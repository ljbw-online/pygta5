import pickle
import os
from multiprocessing import Process, Queue
from time import time, sleep

import cv2
import numpy as np

from common import get_keys, CircularBuffer, resize, put_text
from deep_q.breakout_wrapper import Env, num_actions
from deep_q.model_formats.breakout_frame_seq import create_q_model, frame_shape, model_path, InputCollector, \
    visualise, data_path, frame_seq_len, target_model_path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


def discounted_reward_sums(gamma, rewards):
    rewards_len = len(rewards)

    sum_val = 0
    sums = np.zeros_like(rewards)

    for i in range(rewards_len - 1, -1, -1):
        sums[i] = rewards[i] + gamma * sum_val
        sum_val = sums[i]

    return sums


training_params_filename = data_path + '_training_state'

buffer_filename = data_path + '_replay_buffer.npy'
buffer_dtype = np.dtype([('frames', np.uint8, frame_shape), ('labels', np.float32, (num_actions + 1,))])
# epoch_size = 100
# frame_seq_len = 2

def train_model(inq, outq):
    import tensorflow as tf
    from tensorflow import keras as ks

    max_memory_length, num_epochs, data_count, stage_count = inq.get(block=True)

    sched = ks.optimizers.schedules.PiecewiseConstantDecay([max_memory_length], [1e-4, 1e-4])
    optimizer = ks.optimizers.Adam(learning_rate=sched, clipnorm=1.0)  # clipnorm from Breakout example

    print('Huber loss')
    @tf.function
    def loss_function(y_true, y_pred):
        max_preds = y_true[:, -1]
        q_actions = tf.reduce_sum(tf.multiply(y_true[:, :-1], y_pred), axis=1)
        q_actions = tf.expand_dims(q_actions, axis=1)
        return ks.losses.Huber()(max_preds, q_actions)

    data_len = min(data_count, max_memory_length)
    print(f'training on {data_len} data')

    # MAYBE NEED NP.LOAD WITH MEMMAP ARG
    # with np.memmap(buffer_filename, mode='r', shape=(data_len,), dtype=buffer_dtype) as data:

    data = np.memmap(buffer_filename, mode='r', shape=(data_len,), dtype=buffer_dtype)
    # frames = data['frames'][:(1 - frame_seq_len)]
    # labels = data['labels'][(frame_seq_len - 1):]
    frames = data['frames']
    labels = data['labels']
    # print(len(frames), len(labels))

    data_set = ks.utils.timeseries_dataset_from_array(frames, labels, frame_seq_len)

    # for batch in data_set:
    #     frame_seqs, labels = batch
    #     for frame_seq, label in zip(frame_seqs, labels):
    #         frames = np.hstack(frame_seq)
    #         frames = resize(frames, width=800)
    #         put_text(frames, f'{label.numpy()}')
    #         cv2.imshow('title', frames)
    #         if cv2.waitKey(0) == ord('q'):
    #             cv2.destroyWindow('title')
    #             return

    model = create_q_model()
    model.compile(optimizer=optimizer, loss=loss_function)
    model.set_weights(inq.get(block=True))

    model.fit(data_set, shuffle=True, epochs=num_epochs)

    outq.put(model.get_weights())

    sleep(2)
    if not outq.empty():
        sleep(5)
        print('Subprocess: cancelling join thread')
        outq.cancel_join_thread()

    return


class TrainingParameters:
    def __init__(self):
        self.max_memory_length = 32_000  # 100.8KB for one 210x160x3 frame
        self.max_episode_length = 10_000
        self.random_frames = 0
        self.num_stages = 500
        self.data_per_stage = 32_000
        self.epochs_per_stage = 2
        self.stages_per_target_update = 5

        self.data_count = 0
        self.total_step_count = 0
        self.episode_count = 0
        self.stage_count = 0
        self.data_since_train = 0

        self.gamma = 0.99
        self.epsilon_min = 0.1
        self.epsilon_max = 1.0
        self.game_over_penalty = np.float32(-1)


def train(reuse_models=False):
    import tensorflow as tf
    from tensorflow import keras as ks

    if os.path.isfile(training_params_filename) and 'y' != input('Overwrite training state? (y/N)'):
        with open(training_params_filename, 'rb') as training_params_file:
            tp = pickle.load(training_params_file)

        training_data = np.memmap(buffer_filename, mode='r+', dtype=buffer_dtype, shape=(tp.max_memory_length,))
        training_data = CircularBuffer(training_data)

        model = ks.models.load_model(model_path)
        target_model = ks.models.load_model(target_model_path)

        print('RESUMING')
    else:
        tp = TrainingParameters()

        training_data = np.memmap(buffer_filename, mode='w+', dtype=buffer_dtype, shape=(tp.max_memory_length,))
        training_data = CircularBuffer(training_data)

        if reuse_models:
            print('LOADING MODEL FROM DISC')
            model = ks.models.load_model(model_path)
            target_model = ks.models.load_model(target_model_path)
        else:
            model = create_q_model()
            target_model = create_q_model()

    model.summary()
    print(tp.__dict__)

    env = Env()
    input_collector = InputCollector()

    q_predictions = tf.zeros((num_actions,))

    episode_observations = []
    episode_inputs = []
    episode_actions = []
    episode_rewards = []
    episode_q_predictions = []
    episode_q_actions = []

    image_q = Queue()
    visualisation_process = Process(target=visualise, args=(image_q,))
    visualisation_process.start()

    count_q = Queue()
    weights_q = Queue()

    training_timeout = 1000

    choice = ''
    get_keys()
    while True:
        observation = env.reset()
        model_input = input_collector.get(observation)

        epsilon = max(tp.epsilon_min, tp.epsilon_max - (tp.epsilon_max - tp.epsilon_min) * (tp.stage_count / tp.num_stages))

        episode_observations.clear()
        episode_inputs.clear()
        episode_actions.clear()
        episode_rewards.clear()
        episode_q_predictions.clear()
        episode_q_actions.clear()

        episode_steps = 0

        for _ in range(frame_seq_len - 1):
            episode_observations.append(np.zeros(frame_shape, dtype=np.uint8))

        while True:
            keys = get_keys()

            if tp.total_step_count < tp.random_frames or epsilon > np.random.random(()):
                random_action = True
                action = np.random.choice(num_actions)
            else:
                random_action = False
                input_tensor = tf.convert_to_tensor(model_input)
                input_tensor = tf.expand_dims(input_tensor, 0)
                q_predictions = model(input_tensor, training=False)[0]
                action = tf.argmax(q_predictions).numpy()  # Take best action

            next_observation, reward, terminated = env.step(action, keys)

            episode_observations.append(observation)
            episode_inputs.append(model_input)
            episode_actions.append(action)
            episode_rewards.append(reward)
            episode_q_predictions.append(q_predictions)
            episode_q_actions.append(q_predictions[action])

            if image_q.empty():
                image_q.put((model_input, action, random_action, q_predictions, reward))

            observation = next_observation
            model_input = input_collector.get(observation)

            episode_steps += 1
            tp.total_step_count += 1

            if terminated or (episode_steps == tp.max_episode_length):
                for _ in range(frame_seq_len - 1):
                    episode_rewards.append(0)
                    episode_actions.append(0)

                dr_sums = discounted_reward_sums(tp.gamma, episode_rewards)
                image_q.put((np.array(episode_q_predictions), dr_sums, False))

                # The target_prediction for each step is the future reward as predicted from the *next*
                # model_input. Hence disinclude the first model_input. This means that the first observation from the
                # env is not being used so also disinclude this. The last action and reward values are also not used, by
                # similar reasoning.
                # del episode_observations[0]
                # del episode_inputs[0]
                # del episode_actions[-1]
                # del episode_rewards[-1]

                # if terminated:
                #     episode_target_predictions[-1] = np.ones((num_actions,), dtype=np.float32) * game_over_penalty

                # episode_inputs_tensor = tf.convert_to_tensor(episode_inputs, dtype=tf.uint8)
                episode_inputs_array = np.array(episode_inputs)
                target_predictions = target_model.predict(episode_inputs_array[1:], verbose=0)

                # If we got game-over then manually set future reward prediction for the final observation to -1.
                # It has to be done like this because tensors don't support assignment.
                if terminated:
                    offset = np.zeros((len(episode_inputs) - 1, num_actions), dtype=np.float32)
                    offset[-1] = tp.game_over_penalty - target_predictions[-1]
                    offset = tf.convert_to_tensor(offset)
                    target_predictions = tf.add(target_predictions, offset)

                target_predictions = tf.concat([target_predictions, tf.zeros((frame_seq_len - 1 + 1, num_actions), dtype=tf.float32)], axis=0)

                rewards_tensor = tf.convert_to_tensor(episode_rewards)
                q_values = rewards_tensor + tp.gamma * tf.reduce_max(target_predictions, axis=1)

                # if stage_count % 6 == 0 and stage_count != 0:
                #     for m_input, r, tp in zip(episode_inputs, rewards_tensor, gamma * tf.reduce_max(target_predictions, axis=1)):
                #         fs = np.hstack(m_input)
                #         fs = resize(fs, height=500)
                #         put_text(fs, f'{r.numpy()} {tp.numpy()}')
                #         cv2.imshow('title', fs)
                #         if cv2.waitKey(0) == ord('q'):
                #             cv2.destroyWindow('title')
                #             break
                #
                #     try:
                #         cv2.destroyWindow('title')
                #     except cv2.error:
                #         pass

                labels = np.zeros((len(episode_observations), num_actions + 1), dtype=np.float32)
                labels[:, :-1] = tf.one_hot(episode_actions, num_actions)
                labels[:, -1] = q_values

                assert len(episode_observations) == len(labels)  # Good sanity test to have whilst changing code

                training_data.append(zip(episode_observations, labels))

                episode_length = len(labels)
                tp.data_count += episode_length
                tp.data_since_train += episode_length
                print('data_since_train', tp.data_since_train)

                # tp.episode_count += 1  # only increment if there were enough frames in the episode

                if tp.data_since_train >= tp.data_per_stage:
                    t = time()

                    training_process = Process(target=train_model, args=(count_q, weights_q))
                    training_process.start()

                    count_q.put((tp.max_memory_length, tp.epochs_per_stage, tp.data_count, tp.stage_count))
                    count_q.put(model.get_weights())

                    for i in range(training_timeout):
                        if i == training_timeout - 1:
                            print('Didn\'t get weights from subprocess')
                            break
                        elif not training_process.is_alive():
                            break
                        elif weights_q.empty():
                            sleep(1)
                        else:
                            model.set_weights(weights_q.get(block=True))
                            print('Stage {}, epsilon {:.2f}'.format(tp.stage_count, epsilon))

                            if tp.stage_count % tp.stages_per_target_update == 0 and tp.stage_count > 0:
                                print('Updating target')
                                target_model.set_weights(model.get_weights())
                            break

                    if training_process.is_alive():
                        training_process.join(timeout=10)
                        training_process.close()
                    else:
                        print('training_process died')
                        break

                    # episode_count = 0
                    tp.data_since_train = 0
                    tp.stage_count += 1
                    print('training took {:.1f}s, data_count: {}'.format(time() - t, tp.data_count))
                break

            if ('1' in keys) and ('0' in keys):
                env.pause()
                choice = input('Quit/Save-and-close/Continue (q/s/C)')
                if choice == 'q':
                    image_q.put(None)
                    if visualisation_process.is_alive():
                        visualisation_process.join(timeout=10)

                    env.close()
                    break
                elif choice == 's':
                    training_data.flush()
                    model.save(model_path)
                    target_model.save(target_model_path)

                    # training_state = {
                    #     'max_memory_length': max_memory_length,
                    #     'max_episode_length': max_episode_length,
                    #     'num_stages': num_stages,
                    #     'epochs_per_stage': epochs_per_stage,
                    #     'stages_per_target_update': stages_per_target_update,
                    #     'data_count': data_count,
                    #     'total_step_count': total_step_count,
                    #     'episode_count': episode_count,
                    #     'stage_count': stage_count,
                    #     'data_since_train': data_since_train,
                    #     'data_per_stage': data_per_stage,
                    #     'game_over_penalty': game_over_penalty,
                    #     'gamma': gamma,
                    #     'random_frames': random_frames,
                    #     'epsilon_min': epsilon_min,
                    #     'epsilon_max': epsilon_max
                    # }

                    # tp = TrainingParameters()
                    # tp.max_memory_length = max_memory_length
                    # tp.max_episode_length = max_episode_length
                    # tp.num_stages = num_stages
                    # tp.epochs_per_stage = epochs_per_stage
                    # tp.stages_per_target_update = stages_per_target_update
                    # tp.data_count = data_count
                    # tp.total_step_count = total_step_count
                    # tp.episode_count = episode_count
                    # tp.stage_count = stage_count
                    # tp.data_since_train = data_since_train
                    # tp.data_per_stage = data_per_stage
                    # tp.game_over_penalty = game_over_penalty
                    # tp.gamma = gamma
                    # tp.random_frames = random_frames
                    # tp.epsilon_min = epsilon_min
                    # tp.epsilon_max = epsilon_max

                    with open(training_params_filename, 'wb') as training_params_file:
                        pickle.dump(tp, training_params_file)

                    image_q.put(None)
                    if visualisation_process.is_alive():
                        visualisation_process.join(timeout=10)

                    env.close()
                    break
                else:
                    env.resume()

        if choice == 'q' or choice == 's':
            break


if __name__ == '__main__':
    train()
