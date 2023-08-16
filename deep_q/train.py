import pickle
import os
from collections import deque
from multiprocessing import Process, Queue
from time import time

import numpy as np
import cv2

from common import get_keys, uint8_to_float32
from deep_q.frames_to_3action import create_q_model, Env, num_actions, model_path, data_path, \
    target_model_path, visualise

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

batch_size = 32  # Size of batch taken from replay buffer
update_target_batches = 1000
update_target_frames = batch_size * update_target_batches
gamma = 0.97
max_episode_length = 100


def update_model(inq, outq, resume=False):
    import tensorflow as tf
    from tensorflow import keras as ks
    optimizer = ks.optimizers.Adam(learning_rate=0.00001, clipnorm=1.0)  # clipnorm from Breakout example
    loss_function = ks.losses.Huber()
    batch_count = 1

    if resume:
        # this results in two different models in memory
        model = ks.models.load_model(model_path)
        target_model = ks.models.load_model(model_path)
    else:
        model = create_q_model()
        target_model = create_q_model()

    print_template = 'Updated target_model at batch {}, prediction loss: {:.2f}, regularisation loss {:.2f}'
    while True:
        q_item = inq.get(block=True)

        if q_item is None:
            model.save(model_path)
            target_model.save(target_model_path)

            # If a process has been putting stuff on a queue then Process.join() will hang until everything has been
            # removed from that queue. This method could be called allow_exit_without_flush().
            outq.cancel_join_thread()
            return
        else:
            state_sample, state_next_sample, reward_sample, action_sample, done_sample = q_item

            # Predict Q-values for the sampled future states, using the target model for stability
            future_rewards, _ = target_model.predict(state_next_sample, verbose=0)
            # Q value = reward + discount factor * expected future reward
            discounted_q_values = gamma * tf.reduce_max(future_rewards, axis=1)
            updated_q_values = reward_sample + discounted_q_values
            # updated_q_values = reward_sample + gamma * tf.reduce_max(future_rewards, axis=1)  # definitely a list?

            # updated_q_values contains a -1 whenever "terminated" is True. Otherwise it contains the Q value.
            updated_q_values = updated_q_values * (1 - done_sample) - done_sample

            # Create a mask so we only calculate loss on the updated Q-values
            masks = tf.one_hot(action_sample, num_actions)

            with tf.GradientTape() as tape:
                q_values, _ = model(state_sample)

                # Apply the masks to the Q-values to get the Q-value for action taken
                q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)

                # Calculate loss between new Q-value and old Q-value
                prediction_loss = loss_function(updated_q_values, q_action)
                regularization_loss = sum(model.losses)

                loss = prediction_loss + regularization_loss

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if batch_count % update_target_batches == 0:
                target_model.set_weights(model.get_weights())
                print(print_template.format(batch_count, prediction_loss, regularization_loss))

        batch_count += 1

        outq.put(model.get_weights())


def train(resume=False, reuse_models=False):
    import tensorflow as tf
    from tensorflow import keras as ks

    if resume:
        with open(data_path + '_training_state', 'rb') as training_state_file:
            training_state = pickle.load(training_state_file)

        max_memory_length = training_state['max_memory_length']
        action_history = training_state['action_history']
        # state_history = training_state['state_history']
        # state_next_history = training_state['state_next_history']
        reward_history = training_state['reward_history']
        done_history = training_state['done_history']

        frame_count = training_state['frame_count']
        epsilon = training_state['epsilon']
        epsilon_min = training_state['epsilon_min']
        epsilon_max = training_state['epsilon_max']

        model = ks.models.load_model(model_path)

        env = Env('resume', history_len=max_memory_length)
        print('Resuming training at epsilon == {:.2f}'.format(epsilon))
    else:
        max_memory_length = 50_000  # 14.4KB for one 160x90 greyscale frame
        env = Env('train', history_len=max_memory_length)

        # max_steps_per_episode = 500
        action_history = []
        # state_history = []
        # state_next_history = []
        reward_history = []
        done_history = []
        # episode_reward_history = deque(maxlen=100)  currently don't have episodes

        # episode_count = 0
        frame_count = 0

        epsilon_min = 0.1
        epsilon_max = 0.6
        epsilon = epsilon_max

        if reuse_models:
            model = ks.models.load_model(model_path)
        else:
            model = create_q_model()

    model.summary()
    print('reuse_models: {}, gamma: {}, update_target_batches: {}, max_memory_length {}, max_episode_length {}'
          .format(reuse_models, gamma, update_target_batches, max_memory_length, max_episode_length))

    epsilon_range = (epsilon_max - epsilon_min)
    epsilon_greedy_frames = max_memory_length
    epsilon_decrement = epsilon_range / epsilon_greedy_frames
    # epsilon_random_frames = update_target_frames * 5
    epsilon_random_frames = 0

    batch_q = Queue()
    weights_q = Queue()

    training_process = Process(target=update_model, args=(batch_q, weights_q, resume or reuse_models))
    training_process.start()
    weights_q.put(model.get_weights())

    image_q = Queue()
    visualisation_process = Process(target=visualise, args=(image_q,))
    visualisation_process.start()

    print_template = 'Frame {}, epsilon: {:.2f}, average reward: {:.2f}'

    choice = ''
    get_keys()
    while True:
        # episode_reward = 0
        state = env.reset()
        while True:
            keys = get_keys()
            frame_count += 1

            # Always run model, for visualisation
            state_tensor = tf.convert_to_tensor(state)
            state_tensor = tf.expand_dims(state_tensor, 0)
            [q_predictions], [conv2d] = model(state_tensor, training=False)
            # q_predictions = q_predictions[0]
            # conv2d = conv2d[0]

            if frame_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
                random_action = True
                action = np.random.choice(num_actions)
            else:
                random_action = False
                action = tf.argmax(q_predictions).numpy()  # Take best action

            if frame_count > epsilon_random_frames:  # Only decay epsilon after random_frames has ended
                # epsilon -= epsilon_decrement
                # epsilon = max(epsilon, epsilon_min)
                epsilon = epsilon_max - epsilon_range * (frame_count / epsilon_greedy_frames)
                epsilon = max(epsilon, epsilon_min)

            state_next, reward, terminated = env.step(action, keys)
            # print(reward)

            image_q.put((state, conv2d, action, random_action, q_predictions, reward))

            # episode_reward += reward

            action_history.append(action)
            # state_history.append(state)
            # state_next_history.append(state_next)
            done_history.append(terminated)
            reward_history.append(reward)

            state = state_next

            if not weights_q.empty():
                model.set_weights(weights_q.get())

            if batch_q.empty() and (frame_count % 1 == 0) and len(done_history) > batch_size:
                indices = np.random.choice(range(len(done_history)), size=batch_size)

                state_sample = np.array([env.state_history[i] for i in indices])
                state_next_sample = np.array([env.state_history[i + 1] for i in indices])
                reward_sample = [reward_history[i] for i in indices]
                action_sample = [action_history[i] for i in indices]
                done_sample = tf.convert_to_tensor([float(done_history[i]) for i in indices])

                # ims = []
                # i = 0
                # for stack, next_stack in zip(state_sample, state_next_sample):
                #     ims.append(np.hstack(stack))
                #     ims.append(np.hstack(next_stack))
                #     if np.sum(stack[0]) == 0:
                #         print(len(done_history))
                #         input('zeros frame at index {}'.format(indices[i]))
                #     i += 1
                #
                # ims = np.vstack(ims)
                # cv2.imshow('ims', ims)
                # cv2.waitKey(1)

                batch_q.put((state_sample, state_next_sample, reward_sample, action_sample, done_sample))

            if len(reward_history) > max_memory_length:
                del reward_history[:1]
                # del state_history[:1]
                # del state_next_history[:1]
                del action_history[:1]
                del done_history[:1]

            # print(frame_count, frame_count % update_target_frames)
            if frame_count % 1000 == 0:
                mean_reward = np.mean(reward_history[- update_target_frames:])
                print(print_template.format(frame_count, epsilon, mean_reward))

            if terminated or (frame_count % max_episode_length == 0):
                break

            if ('1' in keys) and ('0' in keys):
                env.pause()
                choice = input('Quit/Save-and-close/Continue (q/s/C)')
                if choice == 'q':
                    batch_q.put(None)
                    training_process.join()
                    image_q.put(None)
                    visualisation_process.join()
                    env.close()
                    break
                elif choice == 's':
                    # Models are saved in the training subprocess
                    training_state = {'epsilon': epsilon, 'max_memory_length': max_memory_length,
                                      'epsilon_min': epsilon_min, 'frame_count': frame_count,
                                      'action_history': action_history, 'done_history': done_history,
                                      'reward_history': reward_history, 'epsilon_max': epsilon_max}

                    with open(data_path + '_training_state', 'wb') as training_state_file:
                        pickle.dump(training_state, training_state_file)

                    env.state_history.save()

                    batch_q.put(None)
                    training_process.join()
                    image_q.put(None)
                    visualisation_process.join()
                    env.close()
                    break
                else:
                    env.resume()

        if choice == 'q' or choice == 's':
            break

        # episode_count += 1


if __name__ == '__main__':
    train(reuse_models=False)
