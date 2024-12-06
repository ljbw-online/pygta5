import os
import pickle
from time import sleep
from collections import deque
from pathlib import Path
from threading import Thread, main_thread
from queue import Queue, Empty
from multiprocessing import Process, Queue
from queue import Empty

import cv2
import numpy as np

from common import get_save_path
from plot import Plot
from q_networks import dueling_architecture as get_q_net
from q_networks import QFunction
from replay_buffer import ReplayBuffer
from environments.gta import Env, gamma, action_labels, env_name

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
import keras
from tensorflow.python.training.checkpoint_management import CheckpointManager
from tensorflow.python.training.tracking.util import Checkpoint

# TensorFlow versions newer than 2.9 seem to have a memory leak somewhere in the model training code, e.g. in model.fit
# and optimizer.apply_gradients

testing = False
if testing:
    from environments.numbers_env import Env, gamma, epsilon_max, action_labels, env_name
    epsilon_interval = 10_000
    steps_per_save = 2_500
    steps_per_evaluation = 1_000
else:
    epsilon_interval = 1_000_000
    steps_per_save = 500_000
    steps_per_evaluation = 500_000

epsilon_max = 1.0
epsilon_min = 0.1
epsilon_range = epsilon_max - epsilon_min
batch_size = 32
steps_per_model_update = 4
steps_per_target_update = 10_000
num_eval_episodes = 30
loss_function = keras.losses.Huber()
optimizer = keras.optimizers.Adam(learning_rate=1e-5, clipnorm=1.0)
gameover_penalty = np.float32(-1.0)

rng = np.random.default_rng()

replay_buffer_dir = get_save_path(env_name, 'replay_buffer')
model_path = get_save_path(env_name, 'model')
target_model_path = get_save_path(env_name, 'target_model')
best_model_path = get_save_path(env_name, 'best_model')
pickleable_state_path = get_save_path(env_name, 'training_state')
checkpoint_dir = get_save_path(env_name, 'checkpoint')


def average_return(num_episodes, environment, q_function):
    total_return = 0
    for _ in range(num_episodes):
        episode, terminated, _, _ = run_episode(environment, q_function, epsilon=environment.evaluation_epsilon)
        total_return += sum(episode['reward']) + np.float32(terminated)

    return total_return / num_episodes


def console_listen(console_q):
    choice = 'n'
    if os.path.isdir(replay_buffer_dir):
        choice = input('Restore from disc? (Y/n)')

    restore = choice != 'n'

    console_q.put(restore)

    while True:
        console_q.put(input())


def discounted_reward_sums(gamma, rewards):
    rewards_len = len(rewards)
    sum_val = 0
    sums = np.zeros_like(rewards)

    for i in range(rewards_len - 1, -1, -1):
        sums[i] = rewards[i] + gamma * sum_val
        sum_val = sums[i]

    return sums


def run_episode(environment, q_function, epsilon=1.0):
    episode = []
    terminated = False
    step_count = 0
    q_predictions = []
    q_prediction_step_counts = []

    q_function.clear()

    observation = environment.reset()
    for _ in range(environment.max_steps_per_episode):
        if terminated:
            break

        if rng.random() < epsilon:
            action = rng.choice(environment.num_actions)
            downsampled_observation = q_function.downsampler(observation)
        else:
            current_step_q_predictions, downsampled_observation = q_function(observation)
            action = tf.argmax(current_step_q_predictions).numpy()
            q_predictions.append(current_step_q_predictions.numpy())
            q_prediction_step_counts.append(step_count)

        observation_next, reward, terminated = environment.step(action)

        # A timestep consists of an observation, the action we took in response to that observation,
        # and the reward the environment gave us in response to that action.
        episode.append(np.array((downsampled_observation, action, reward), dtype=environment.timestep_dtype))

        observation = observation_next
        step_count += 1

    # Release controls in GTA
    environment.pause()

    return np.array(episode), terminated, q_predictions, q_prediction_step_counts


def display_plot(inq, outq):
    plot = Plot()
    plot.title = 'Q-values'
    plot.xlabel = 'Step count'
    plot.width = 12
    key = 0

    window_name = 'Q-values'
    cv2.namedWindow(window_name)

    rewards, q_predictions, q_prediction_step_counts = inq.get()
    while True:
        try:
            rewards, q_predictions, q_prediction_step_counts  = inq.get(block=False)
        except Empty:
            pass
        
        plot.clear()
        plot.add_line(range(len(rewards)), discounted_reward_sums(gamma, rewards), label='discounted reward')
        
        for label, action_values in zip(action_labels, np.transpose(q_predictions)):
            plot.add_line(q_prediction_step_counts, action_values, label=label)

        cv2.imshow(window_name, plot.to_array())
        key_ord = cv2.waitKey(33)

        if key_ord != -1:
            key = chr(key_ord)
            outq.put(key)

        # if key == 'q':
        #     break

        

class PickleableTrainingState:
    def __init__(self, replay_buffer_dir, env, obs_seq_len, max_episodes):
        self.step_count = 0
        self.best_average_return = 0
        self.episode_returns = deque(maxlen=100)
        self.steps_since_target_update = 0
        self.steps_since_save = 0
        self.steps_since_evaluation = 0

        if testing:
            self.replay_buffer = ReplayBuffer(replay_buffer_dir, env.name, env.timestep_dtype,
                                              obs_seq_len=obs_seq_len, max_episodes=10)
        else:
            self.replay_buffer = ReplayBuffer(replay_buffer_dir, env.name, env.timestep_dtype, 
                                              obs_seq_len=obs_seq_len)


class TrainingState:
    def __init__(self, env, obs_seq_len=4, max_episodes=1_000, restore=False):
        checkpoint = Checkpoint(optimizer=optimizer)
        self.checkpoint_manager = CheckpointManager(checkpoint, checkpoint_dir, 1)

        if restore:
            with open(pickleable_state_path, 'rb') as pickleable_state_file:
                self.pts = pickle.load(pickleable_state_file)

            # Re-create all of the memory maps
            self.pts.replay_buffer.populate_episodes_deque()

            self.model = keras.models.load_model(model_path)
            self.target_model = keras.models.load_model(target_model_path)

            checkpoint.restore(self.checkpoint_manager.latest_checkpoint)

        else:
            # Moved this to replay_buffer.py
            # try:
            #     os.makedirs(replay_buffer_dir)
            # except FileExistsError:
            #     pass

            self.pts = PickleableTrainingState(replay_buffer_dir, env, obs_seq_len, max_episodes)

            observation_shape = env.timestep_dtype['observation'].shape
            input_shape = observation_shape + (obs_seq_len,)
            self.model = get_q_net(input_shape, env.num_actions)
            self.target_model = get_q_net(input_shape, env.num_actions)

    def save(self):
        print('Saving training state...')

        # We've seen "OSError: too many open files" when saving. Maybe because this:
        # self.pts.replay_buffer.episodes.clear()
        # doesn't actually cause all of the memory maps to get garbage collected.
        
        # Stop all of the episode arrays from being included in the pickle file
        episodes = self.pts.replay_buffer.episodes
        self.pts.replay_buffer.episodes = deque()

        with open(pickleable_state_path, 'wb') as pickleable_state_file:
            pickle.dump(self.pts, pickleable_state_file)

        self.model.save(model_path)
        self.target_model.save(target_model_path)

        self.checkpoint_manager.save()

        # Add all of the memmaps back in case we want to continue training
        self.pts.replay_buffer.episodes = episodes
        # episodes = None
        
        print('... Done.')


def main():
    console_queue = Queue()
    
    # daemon allows the script to exit whilst this thread is running.
    # This thread has caused Python to crash before, possibly because I was calling the input function 
    # in the main thread as well.
    thread = Thread(target=console_listen, args=(console_queue,), daemon=True)

    key = 0
    inq = Queue()
    outq = Queue()
    process = Process(target=display_plot, args=(inq, outq), daemon=True)
    process.start()

    env = Env()

    obs_seq_length = 4

    sleep(1)  # OpenCV warnings

    thread.start()

    restore = console_queue.get()

    if restore:
        print('Restoring')

    ts = TrainingState(env, restore=restore)

    q_function = QFunction(obs_seq_length, env, ts.model)

    # Stops a warning when saving
    ts.model.compile()
    ts.target_model.compile()

    print(f'len(replay_buffer): {len(ts.pts.replay_buffer)}')

    while True:
        try:
            key = console_queue.get(block=False)
        except Empty:
            pass
        
        match key:
            case 's':
                ts.save()
            case 'q':
                print('Save? (Y/n)')
                save_choice  = console_queue.get()

                # Not saving here stops restoring from working later due to reply buffer implementation
                if save_choice  != 'n':
                    ts.save()

                env.close()
                cv2.destroyAllWindows()
                break

        epsilon = max(epsilon_max - epsilon_range * (ts.pts.step_count / epsilon_interval), epsilon_min)
        episode, terminated, q_predictions, q_prediction_step_counts = run_episode(env, q_function, epsilon)

        # clip rewards
        for i, reward in enumerate(episode['reward']):
            episode['reward'][i] = min(1.0, reward)

        if terminated:
            episode['reward'][-1] = gameover_penalty

        ts.pts.replay_buffer.add_episode(episode)

        ts.pts.step_count += len(episode)
        ts.pts.steps_since_target_update += len(episode)
        ts.pts.steps_since_save += len(episode)
        ts.pts.steps_since_evaluation += len(episode)
        ts.pts.episode_returns.append(sum(episode['reward']) + np.float32(terminated))

        inq.put((episode['reward'], q_predictions, q_prediction_step_counts))

        try:
            key = outq.get(block=False)
        except Empty:
            key = ''

        # Update every fourth frame
        for _ in range(int(len(episode) / steps_per_model_update)):
        # for _ in range(len(episode)):  # Update once for every timestep collected
            observation_sample, observation_next_sample, action_sample, reward_sample = ts.pts.replay_buffer.get_batch()

            # (batch, seq_len, width, height) -> (batch, width, height, seq_len)
            observation_sample = np.moveaxis(observation_sample, 1, -1)
            observation_next_sample = np.moveaxis(observation_next_sample, 1, -1)

            future_actions = tf.argmax(ts.model.predict(observation_next_sample, verbose=0), axis=1)
            future_onehot_actions = tf.one_hot(future_actions, env.num_actions)

            future_value_estimates = ts.target_model.predict(observation_next_sample, verbose=0)
            future_action_values = tf.reduce_sum(tf.multiply(future_value_estimates, future_onehot_actions), axis=1)

            updated_q_values = reward_sample + gamma * future_action_values

            onehot_actions = tf.one_hot(action_sample, env.num_actions)

            with tf.GradientTape() as tape:
                q_values = ts.model(observation_sample)
                q_action = tf.reduce_sum(tf.multiply(q_values, onehot_actions), axis=1)
                loss = loss_function(updated_q_values, q_action)

            grads = tape.gradient(loss, ts.model.trainable_variables)
            optimizer.apply_gradients(zip(grads, ts.model.trainable_variables))

        print(f'step_count: {ts.pts.step_count}, loss: {loss}, average return: {np.mean(ts.pts.episode_returns)}')

        if ts.pts.steps_since_target_update >= steps_per_target_update:
            # If we set steps_since_target_update to zero then the updates go out of phase with multiples of
            # steps_per_target_update
            ts.pts.steps_since_target_update = ts.pts.steps_since_target_update - steps_per_target_update
            ts.target_model.set_weights(ts.model.get_weights())

        if ts.pts.steps_since_save >= steps_per_save:
            ts.pts.steps_since_save = ts.pts.steps_since_save - steps_per_save
            ts.save()

        if ts.pts.steps_since_evaluation >= steps_per_evaluation:
            ts.pts.steps_since_evaluation = ts.pts.steps_since_evaluation - steps_per_evaluation
            evaluation_return = average_return(30, env, q_function)

            print(f'evaluation_return: {evaluation_return}')
            if evaluation_return > ts.pts.best_average_return:
                ts.pts.best_average_return = evaluation_return
                print('Saving best model')
                ts.model.save(best_model_path)


if __name__ == '__main__':
    main()
