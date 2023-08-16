import os
from multiprocessing import Process, Queue
import pickle

import cv2
import numpy as np

from common import get_keys, resize, put_text
from deep_q.breakout_wrapper import Env, num_actions
from deep_q.train_in_stages import discounted_reward_sums, training_params_filename, TrainingParameters
from model_formats.breakout_frame_seq import model_path, visualise, InputCollector, data_path
# from model_formats.breakout_single_frame import data_path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


def main(save_vis=False, single_step=False):
    import tensorflow as tf
    from tensorflow import keras as ks

    with open(training_params_filename, 'rb') as training_params_file:
        tp = pickle.load(training_params_file)

    # gamma = training_state['gamma']
    # max_episode_length = training_state['max_episode_length']

    model = ks.models.load_model(model_path, custom_objects={'loss_function': None})

    env = Env()
    input_collector = InputCollector()

    frame_count = 0
    episode_rewards = []
    episode_q_predictions = []
    episode_q_actions = []

    episode_reward_sums =[]

    image_q = Queue()
    visualisation_process = Process(target=visualise, args=(image_q, save_vis))
    if not single_step:
        visualisation_process.start()

    # fig, ax = plt.subplots(figsize=(7, 4))
    # [predicted_plot] = ax.plot(0, 0, label='predicted')
    # [actual_plot] = ax.plot(0, 0, label='actual')
    # ax.set_title('Q-prediction performance')
    # ax.set_xlabel('timestep')
    # ax.set_ylabel('discounted reward sum')
    # ax.set_ylim((-1, 2))
    # ax.legend()
    # plt.show(block=False)
    save_choice = False

    get_keys()
    while True:
        observation = env.reset()
        model_input = input_collector.get(observation)

        episode_rewards.clear()
        episode_q_predictions.clear()
        episode_q_actions.clear()
        while True:
            keys = get_keys()
            frame_count += 1

            state_tensor = tf.convert_to_tensor(model_input)
            state_tensor = tf.expand_dims(state_tensor, 0)
            # vis_output = vis_model(state_tensor, training=False)
            # q_values = vis_output[0][0].numpy()[0]
            # conv2d = vis_output[1][0]
            q_values = model(state_tensor)[0]

            random_action = False
            action = tf.argmax(q_values).numpy()  # Take best action

            episode_q_predictions.append(q_values)
            episode_q_actions.append(q_values[action])

            if 0.1 > np.random.rand(1)[0]:
                random_action = True
                action = np.random.choice(num_actions)

            # the puck stays in contact with a brick for two frames. the first frame in which the brick is no longer
            # there is the frame on which we get a reward of 1.0.
            next_observation, reward, terminated = env.step(action, keys)

            if single_step:
                image = resize(next_observation, height=500)
                put_text(image, f'{reward}')
                put_text(image, f'{q_values.numpy()}', size=0.6, position=(50, 100), thickness=1)
                cv2.imshow('title', image)
                cv2.waitKey(0)
            else:
                image_q.put((model_input, action, random_action, q_values, reward))

            episode_rewards.append(reward)
            observation = next_observation
            model_input = input_collector.get(observation)

            if terminated or frame_count == tp.max_episode_length:
                dr_sums = discounted_reward_sums(tp.gamma, episode_rewards)

                if save_vis:
                    save_choice = 'y' == input('Save visualisations? (y/N)')

                # np.array necessary for list of tensors to survive being pickled apparently
                image_q.put((np.array(episode_q_predictions), dr_sums, save_choice))

                break

            if '5' in keys:
                cv2.destroyAllWindows()
                break

        episode_reward_sums.append(sum(episode_rewards))
        print('Average reward: {:.2f}'.format(sum(episode_reward_sums) / len(episode_reward_sums)))

        if '5' in keys:
            cv2.destroyAllWindows()
            image_q.put(None)
            image_q.cancel_join_thread()
            if visualisation_process.is_alive():
                visualisation_process.join()
            env.close()
            break


if __name__ == '__main__':
    main()
