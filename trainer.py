import json
import os
import pickle
from time import sleep
from collections import deque
from threading import Thread
from queue import Empty
from multiprocessing import Process, Queue

import cv2
import numpy as np
from websockets.sync.server import serve

from common import get_save_path
from plot import Plot
from q_networks import QFunction
from q_networks import dueling_architecture as get_q_net
from replay_buffer import ReplayBuffer
from environments.breakout import Env, gamma, action_labels, env_name

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
import keras
from tensorflow.train import Checkpoint, CheckpointManager
# from tensorflow.python.training.checkpoint_management import CheckpointManager
# from tensorflow.python.training.tracking.util import Checkpoint

# TensorFlow versions newer than 2.9 seem to have a memory leak somewhere in the model training code, e.g. in model.fit
# and optimizer.apply_gradients

rng = np.random.default_rng()
websocket_port = 7001

testing = True
if testing:
    from environments.numbers_env import Env, gamma, epsilon_max, action_labels, env_name
    epsilon_interval = 2_500
    iterations_per_save = 500
    iterations_per_evaluation = 250
    num_eval_episodes = 3
    evaluation_epsilon = 1.0
else:
    epsilon_max = 1.0
    epsilon_interval = 250_000
    iterations_per_save = 125_000
    iterations_per_evaluation = 125_000
    num_eval_episodes = 30
    evaluation_epsilon = 0.05

epsilon_min = 0.1
epsilon_range = epsilon_max - epsilon_min
batch_size = 32
steps_per_model_update = 2 / 3  # 3 updates for 2 timesteps
iterations_per_target_update = 10_000
loss_function = keras.losses.Huber()
optimizer = keras.optimizers.Adam(learning_rate=1e-5, clipnorm=1.0)
gameover_penalty = np.float32(-1.0)
obs_seq_len = 4

replay_buffer_dir = get_save_path(env_name, 'replay_buffer')
model_path = get_save_path(env_name, 'model.keras')
target_model_path = get_save_path(env_name, 'target_model.keras')
best_model_path = get_save_path(env_name, 'best_model.keras')
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


def display_plot(inq):
    plot = Plot()
    plot.title = 'Q-values'
    plot.xlabel = 'Step count'
    plot.width = 12

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
        cv2.waitKey(33)


def test_handler(websock):
    print(f'websock received {type(websock.recv())}')


def run_server(inq, outq):
    print('hello from server process')
    # agent_comms has to refer to inq and outq but we can't pass them in as an arguments.
    def agent_comms(websock):
        while True:
            try:
                trainer_msg = inq.get(block=False)
                websock.send(trainer_msg)
            except Empty:
                try:
                    # Use a recv timeout so that keep checking for things to send
                    outq.put(websock.recv(1))
                except TimeoutError:
                    pass

    with serve(agent_comms, "localhost", websocket_port, max_size=None) as server:
        print('starting server')
        server.serve_forever()


def send_json_to_agent(ts, server_in_q, currently_evaluating=False):
    if currently_evaluating:
        epsilon = evaluation_epsilon
    else:
        epsilon = max(epsilon_max - epsilon_range * (ts.pts.timestep_count / epsilon_interval), epsilon_min)

    weights_config = keras.saving.serialize_keras_object(ts.model.get_weights())

    data_for_agent = {'epsilon': epsilon, 'weights_config': weights_config}
    json_for_agent = json.dumps(data_for_agent)

    server_in_q.put(json_for_agent)


def handle_new_episode(ts, episode, terminated, q_predictions, q_prediction_step_counts, display_q):
    # ALL OF THIS COULD BE A TrainingState METHOD
    # clip rewards
    for i, reward in enumerate(episode['reward']):
        episode['reward'][i] = min(1.0, reward)

    if terminated:
        episode['reward'][-1] = gameover_penalty

    ts.pts.replay_buffer.add_episode(episode)

    ts.pts.timestep_count += len(episode)
    # ts.pts.timesteps_since_target_update += len(episode)
    # ts.pts.timesteps_since_save += len(episode)
    # ts.pts.timesteps_since_evaluation += len(episode)

    episode_return = sum(episode['reward']) + np.float32(terminated)

    ts.pts.episode_returns.append(episode_return)
    # ----

    display_q.put((episode['reward'], q_predictions, q_prediction_step_counts))

    return episode_return


class PickleableTrainingState:
    def __init__(self, replay_buffer_dir, env, obs_seq_len, max_episodes):
        self.timestep_count = 0
        self.best_average_return = 0
        self.episode_returns = deque(maxlen=100)
        self.iterations_since_target_update = 0
        self.iterations_since_save = 0
        self.iterations_since_evaluation = 0
        self.training_iteration_count = 0

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
            # pickleable_state_file : SupportsWrite[bytes]
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

    display_q = Queue()
    display_process = Process(target=display_plot, args=(display_q,), daemon=True)
    display_process.start()

    server_in_q = Queue()
    server_out_q = Queue()
    server_process = Process(target=run_server, args=(server_in_q, server_out_q), daemon=True)

    env = Env()

    # Wait for OpenCV warnings before starting console_listen thread
    sleep(1)
    thread.start()

    restore = console_queue.get()

    if restore:
        print('Restoring')

    ts = TrainingState(env, restore=restore)

    # Stops a warning when saving
    ts.model.compile()
    ts.target_model.compile()

    print(f'len(replay_buffer): {len(ts.pts.replay_buffer)}')

    print('sending first msg to agent')
    send_json_to_agent(ts, server_in_q)
    send_json_to_agent(ts, server_in_q)  # Send weights to agent again to stop it from waiting indefinitely.

    # Server has to be started after we've put the first thing on the queue otherwise it will wait indefinitely.
    server_process.start()

    print('waiting for first episode to come back')
    # Wait for the first episode to come back so that we have something to train with.
    json_from_agent = server_out_q.get(block=True)
    print('received first episode')

    episode, terminated, q_predictions, q_prediction_step_counts = json.loads(json_from_agent)

    for i, timestep_list in enumerate(episode):
        episode[i] = tuple(timestep_list)

    episode = np.array(episode, dtype=env.timestep_dtype)

    handle_new_episode(ts, episode, terminated, q_predictions, q_prediction_step_counts, display_q)

    currently_evaluating = False
    evaluation_returns = []
    evaluation_episode_count = 0
    key = ''

    while True:
        try:
            key = console_queue.get(block=False)
        except Empty:
            pass

        match key:
            case 's':
                ts.save()
                key = ''  # Reset so we don't keep saving
            case 'q':
                print('Save? (Y/n)')
                save_choice  = console_queue.get()

                # Not saving here stops restoring from working later due to the replay buffer implementation
                if save_choice  != 'n':
                    ts.save()

                env.close()
                cv2.destroyAllWindows()
                break

        try:
            # episode, terminated, q_predictions, q_prediction_step_counts = run_episode(env, q_function, epsilon)
            episode, terminated, q_predictions, q_prediction_step_counts = json.loads(server_out_q.get(block=False))

            for i, timestep_list in enumerate(episode):
                episode[i] = tuple(timestep_list)

            episode = np.array(episode, dtype=env.timestep_dtype)

            new_episode = True
        except Empty:
            print('no new ep')
            new_episode = False

        if new_episode:
            episode_return = handle_new_episode(ts, episode, terminated, q_predictions, q_prediction_step_counts,
                                                display_q)

            if currently_evaluating:
                evaluation_returns.append(episode_return)
                evaluation_episode_count += 1

            print(
                f'training_iteration_count: {ts.pts.training_iteration_count}, loss: {loss}, '
                f'average return: {np.mean(ts.pts.episode_returns)}, timesteps collected: {ts.pts.timestep_count}')

            send_json_to_agent(ts, server_in_q, currently_evaluating)

        # Update the model once for every steps_per_model_update timesteps collected
        for _ in range(int(len(episode) / steps_per_model_update)):
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

            ts.pts.training_iteration_count += 1
            ts.pts.iterations_since_target_update += 1
            ts.pts.iterations_since_save += 1
            ts.pts.iterations_since_evaluation += 1

        # Setting the steps_since_* variable to zero causes us to go out of phase with multiples of the
        # corresponding steps_per_* variable.
        if ts.pts.iterations_since_target_update >= iterations_per_target_update:
            ts.pts.iterations_since_target_update = ts.pts.iterations_since_target_update - iterations_per_target_update
            ts.target_model.set_weights(ts.model.get_weights())

        if ts.pts.iterations_since_save >= iterations_per_save:
            ts.pts.iterations_since_save = ts.pts.iterations_since_save - iterations_per_save
            ts.save()

        if ts.pts.iterations_since_evaluation >= iterations_per_evaluation:
            ts.pts.iterations_since_evaluation = ts.pts.iterations_since_evaluation - iterations_per_evaluation
            currently_evaluating = True
            print('Starting evaluation')

        if evaluation_episode_count == num_eval_episodes:
            evaluation_return = np.mean(evaluation_returns)

            currently_evaluating = False
            evaluation_episode_count = 0
            evaluation_returns.clear()

            print(f'Evaluation average return: {evaluation_return}')
            if evaluation_return > ts.pts.best_average_return:
                ts.pts.best_average_return = evaluation_return
                print('Saving best model')
                ts.model.save(best_model_path)


if __name__ == '__main__':
    main()
