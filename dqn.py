import os
import pickle
from collections import deque
from threading import Thread

import cv2
import numpy as np

from common import model_dir, data_dir, checkpoint_dir
from q_networks import dueling_architecture as get_q_net
from environments.breakout import Env, gamma, epsilon_max, max_steps_per_episode
from plot import Plot

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
import keras
from tensorflow.python.training.checkpoint_management import CheckpointManager
from tensorflow.python.training.tracking.util import Checkpoint

# TensorFlow versions newer than 2.9 seem to have a memory leak somewhere in the model training code, e.g. model.fit
# or optimizer.apply_gradients

epsilon_min = 0.1
epsilon_range = epsilon_max - epsilon_min
epsilon_interval = 1_000_000
batch_size = 32
steps_per_model_update = 4
steps_per_target_update = 10_000
max_replay_buffer_length = 250_000
episodes_per_evaluation = 100
loss_function = keras.losses.Huber()
optimizer = keras.optimizers.Adam(learning_rate=1e-5, clipnorm=1.0)
gameover_penalty = np.float32(-1.0)

rng = np.random.default_rng()

env = Env()

model_path = os.path.join(model_dir, env.name)
target_model_path = os.path.join(model_dir, env.name + '_target')
best_model_path = os.path.join(model_dir, env.name + '_best')
replay_buffer_path = os.path.join(data_dir, env.name + '_replay_buffer')
training_state_path = os.path.join(data_dir, env.name + '_training_state')
checkpoint_dir = os.path.join(checkpoint_dir, env.name)

checkpoint = Checkpoint(optimizer=optimizer)
checkpoint_manager = CheckpointManager(checkpoint, checkpoint_dir, 1)

observation_window_title = 'Observation'
episode_q_predictions = []
episode_prediction_step_counts = []
episode_rewards = []

labels = ['W', 'no-keys', 'S', 'WA', 'WD']


def compute_average_return(environment, model, num_episodes=10, single_step=False, render_env=False, render_obs=False):
    total_return = 0.0

    if single_step:
        wait_key_duration = 0
    else:
        wait_key_duration = 1

    # When we lose a life in Breakout we need to press action 1 to start the next life. Often the agent doesn't choose
    # action 1 in response to the lost-life screen and hence gets indefinitely stuck. To work around this, if the env
    # has a random_eval_action we inject it with a low probability.
    try:
        random_eval_action = environment.random_eval_action
    except AttributeError:
        random_eval_action = None

    for _ in range(num_episodes):
        observation = environment.reset()
        terminated = False
        episode_return = 0.0

        for _ in range(max_steps_per_episode):
            if terminated:
                break

            if render_obs:
                cv2.imshow(observation_window_title, np.hstack(np.split(observation, 4, axis=2)))
                cv2.waitKey(1)

            if single_step or render_env:
                key = environment.render()
                # if key == ord('q'):
                #     env.close()
                #     return

            # if cv2.waitKey(wait_key_duration) == ord('q'):
            #     cv2.destroyAllWindows()
            #     return None

            q_predictions = model(tf.convert_to_tensor(np.expand_dims(observation, axis=0)), training=False)
            action = tf.argmax(q_predictions[0]).numpy()

            if random_eval_action is not None:
                if rng.random() < 0.05:
                    action = rng.choice(environment.num_actions)

            observation, reward, terminated = environment.step(action)

            if not terminated:
                episode_return += reward

        if single_step:
            print(f'Episode return: {episode_return}')

        total_return += episode_return

    avg_return_local = total_return / num_episodes
    return avg_return_local


input_string = ''


def get_input():
    global input_string
    while True:
        input_string = input()
        if input_string == 'quit' or input_string == 'save_and_quit':
            break


def discounted_reward_sums(gamma, rewards):
    rewards_len = len(rewards)
    sum_val = 0
    sums = np.zeros_like(rewards)

    for i in range(rewards_len - 1, -1, -1):
        sums[i] = rewards[i] + gamma * sum_val
        sum_val = sums[i]

    return sums


class TrainingState:
    def __init__(self):
        self.step_count = 0
        self.best_eval_return = 0
        self.episode_count = 0
        self.episode_returns = deque(maxlen=100)


def main():
    global input_string
    thread = Thread(target=get_input)

    plot = Plot()
    plot.title = 'Q-values'
    plot.xlabel = 'Step count'
    plot.width = 12
    plot.top = 1.2
    plot.bottom = gameover_penalty - 0.1

    choice = 'n'
    if os.path.isfile(replay_buffer_path):
        choice = input('Restore from disc? (y/N)')

    if choice == 'y':
        model = keras.models.load_model(model_path)
        target_model = keras.models.load_model(target_model_path)

        checkpoint.restore(checkpoint_manager.latest_checkpoint)

        with open(replay_buffer_path, 'rb') as data_file:
            replay_buffer = pickle.load(data_file)

        with open(training_state_path, 'rb') as training_state_file:
            ts = pickle.load(training_state_file)
    else:
        model = get_q_net(env.input_shape, env.num_actions)
        target_model = get_q_net(env.input_shape, env.num_actions)

        replay_buffer = deque(maxlen=max_replay_buffer_length)
        ts = TrainingState()

    thread.start()

    # Stops a warning when saving
    model.compile()
    target_model.compile()

    print(f'len(replay_buffer): {len(replay_buffer)}')

    def save_training_state():
        print('Saving models, optimiser, replay buffer and training state.')
        model.save(model_path)
        target_model.save(target_model_path)

        checkpoint_manager.save()

        with open(replay_buffer_path, 'wb') as data_file:
            pickle.dump(replay_buffer, data_file)

        with open(training_state_path, 'wb') as training_state_file:
            pickle.dump(ts, training_state_file)

        print('Saved.')

    terminated = False
    observation = env.reset()

    # Put a batch of timesteps into the replay buffer. The +1 is because we always need an observation_next.
    for _ in range(batch_size + 1):
        if terminated:
            observation = env.reset()

        action = rng.choice(env.num_actions)
        observation_next, reward, terminated = env.step(action)
        replay_buffer.append(np.array((observation, action, reward), dtype=env.timestep_dtype))
        observation = observation_next

    while True:
        if input_string == 'quit':
            env.close()
            cv2.destroyAllWindows()
            thread.join()
            break
        elif input_string == 'save_and_quit':
            save_training_state()
            env.close()
            cv2.destroyAllWindows()
            thread.join()
            break
        elif input_string == 'save':
            save_training_state()
            input_string = ''

        observation = env.reset()
        episode_return = 0
        episode_step_count = 0
        episode_q_predictions.clear()
        episode_prediction_step_counts.clear()
        episode_rewards.clear()

        for _ in range(max_steps_per_episode):
            if input_string == 'quit':
                break

            if terminated:
                terminated = False
                break

            ts.step_count += 1
            episode_step_count += 1

            # env.render()
            # cv2.imshow(observation_window_title, np.hstack(np.split(observation, 4, axis=2)))
            # cv2.waitKey(1)

            if rng.random() < max(epsilon_min, epsilon_max - epsilon_range * (ts.step_count / epsilon_interval)):
                action = rng.choice(env.num_actions)
            else:
                model_input = np.expand_dims(observation, axis=0)
                model_input = tf.convert_to_tensor(model_input)
                q_predictions = model(model_input, training=False)[0]
                action = tf.argmax(q_predictions).numpy()
                episode_q_predictions.append(q_predictions.numpy())
                episode_prediction_step_counts.append(episode_step_count)

            observation_next, reward, terminated = env.step(action)

            episode_return += reward

            if terminated:
                reward = gameover_penalty

            episode_rewards.append(reward)

            # A timestep consists of an observation, the action we took in response to it, and the reward we got on
            # the next step.
            replay_buffer.append(np.array((observation, action, reward), dtype=env.timestep_dtype))

            observation = observation_next

            # Update every fourth frame
            if ts.step_count % steps_per_model_update == 0:
                # The -1 is because we get replay_buffer[i + 1] below
                indices = rng.choice(range(len(replay_buffer) - 1), size=batch_size)

                observation_sample = np.array([replay_buffer[i]['observation'] for i in indices])
                observation_next_sample = np.array([replay_buffer[i + 1]['observation'] for i in indices])
                action_sample = [replay_buffer[i]['action'] for i in indices]
                reward_sample = [replay_buffer[i]['reward'] for i in indices]

                future_actions = tf.argmax(model.predict(observation_next_sample, verbose=0), axis=1)
                future_onehot_actions = tf.one_hot(future_actions, env.num_actions)

                future_value_estimates = target_model.predict(observation_next_sample, verbose=0)
                future_action_values = tf.reduce_sum(tf.multiply(future_value_estimates, future_onehot_actions), axis=1)

                updated_q_values = reward_sample + gamma * future_action_values

                onehot_actions = tf.one_hot(action_sample, env.num_actions)

                with tf.GradientTape() as tape:
                    q_values = model(observation_sample)
                    q_action = tf.reduce_sum(tf.multiply(q_values, onehot_actions), axis=1)
                    loss = loss_function(updated_q_values, q_action)

                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if ts.step_count % steps_per_target_update == 0:
                target_model.set_weights(model.get_weights())

            if ts.step_count % max_replay_buffer_length == 0:
                save_training_state()

        ts.episode_count += 1
        ts.episode_returns.append(episode_return)

        plot.clear()
        plot.add_line(range(len(episode_rewards)), discounted_reward_sums(gamma, episode_rewards),
                      label='discounted reward')

        for label, action_values in zip(labels, np.transpose(episode_q_predictions)):
            plot.add_line(episode_prediction_step_counts, action_values, label=label)

        cv2.imshow('Q-values', plot.to_array())
        cv2.waitKey(1)

        if ts.episode_count % 1 == 0:
            print(f'step_count: {ts.step_count}, loss: {loss}, '
                  f'average return: {np.mean(ts.episode_returns)}')

        if ts.episode_count % episodes_per_evaluation == 0:
            eval_return = compute_average_return(env, model, num_episodes=10, render_env=False, render_obs=False)
            print(f'eval_return: {eval_return}')
            if eval_return > ts.best_eval_return:
                ts.best_eval_return = eval_return

                save_training_state()

                print("Saving best model.")
                model.save(best_model_path)


if __name__ == '__main__':
    main()
