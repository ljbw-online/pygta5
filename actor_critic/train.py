import os
from random import randrange
from socket import socket
# from threading import Thread
from time import sleep, time
from queue import SimpleQueue

import numpy as np

# I think this doesn't work because a relative import doesn't work when you're running a script which is inside
# the package directory:
# from . import __name__ as format_name
from actor_critic import resume_bytes, set_controls, position_bytes, pause_bytes, model_file_name
from common import get_keys, MODEL_DIR, reinforcement_resumed_bytes, reinforcement_paused_bytes

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
from tensorflow import keras as ks

np.set_printoptions(precision=3, floatmode='fixed', suppress=True, sign=' ')
rng = np.random.default_rng().integers

eps = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0
gamma = 1.0
num_actions = 2
sub_episode_timesteps = 60
episode_timesteps = sub_episode_timesteps * 2

l2 = ks.regularizers.l2(1.0)

inputs = ks.layers.Input(shape=(2,))
common1 = ks.layers.Dense(32, activation='relu')(inputs)
common2 = ks.layers.Dense(32, activation='relu')(common1)
action = ks.layers.Dense(num_actions, activation='softmax')(common2)
critic = ks.layers.Dense(1)(common2)

model = ks.Model(inputs=inputs, outputs=[action, critic])

optimizer = ks.optimizers.Adam(learning_rate=0.001)
huber_loss = ks.losses.Huber()
action_probs_history = []
critic_value_history = []
rewards_history = []
running_reward = 0
episode_count = 0

starting_position = np.array([1098.8, -265.4, 68.86], dtype=np.float32)
starting_heading = np.float32([0])
teleport_bytes = bytes([19, 0, 0, 0]) + starting_position.tobytes() + bytes([0])
heading_bytes = bytes([20, 0, 0, 0]) + np.array([0, 0, 0], dtype=np.float32).tobytes() + bytes([0])  # heading == 0


def dist(v2, v1):
    return np.sqrt(sum((v2 - v1)**2))


class InputCollector:
    def __init__(self):
        self.current_position = starting_position.copy()
        self.current_heading = starting_heading.copy()

        self.timestep = 0

        self.one_sec_time = time()
        self.one_sec_position = starting_position.copy()
        self.five_sec_time = time()
        self.five_sec_position = starting_position.copy()
        self.ten_sec_time = time()
        self.ten_sec_position = starting_position.copy()

        self.one_sec_q = SimpleQueue()
        self.five_sec_q = SimpleQueue()
        self.ten_sec_q = SimpleQueue()

        self.sock = socket()
        while True:
            try:
                self.sock.connect(("127.0.0.1", 7001))
                break
            except ConnectionRefusedError:
                print("ConnectionRefusedError")
                sleep(1)

    def return_inputs(self):
        start_time = time()
        while start_time - self.one_sec_time > 1.2:
            if not self.one_sec_q.empty():  # happens if we've been paused (via Z) for a while
                self.five_sec_q.put((self.one_sec_time, self.one_sec_position))
                self.one_sec_time, self.one_sec_position = self.one_sec_q.get()
            else:
                break

        while start_time - self.five_sec_time > 5.2:
            if not self.five_sec_q.empty():
                # self.ten_sec_q.put((self.five_sec_time, self.five_sec_position))
                self.five_sec_time, self.five_sec_position = self.five_sec_q.get()
            else:
                break

        # while start_time - self.ten_sec_time > 10.2:
        #     if not self.ten_sec_q.empty():
        #     self.ten_sec_time, self.ten_sec_position = self.ten_sec_q.get()

        disp = self.current_position - starting_position
        one_sec_displacement = self.current_position - self.one_sec_position
        five_sec_displacement = self.current_position - self.five_sec_position
        # ten_sec_displacement = self.current_position - self.ten_sec_position

        ts = np.array([self.timestep / episode_timesteps], dtype=np.float32)
        self.timestep = (self.timestep + 1) % episode_timesteps

        disp = disp / sub_episode_timesteps
        one_sec_displacement = one_sec_displacement / sub_episode_timesteps

        displacements = np.concatenate((ts, self.current_heading))

        state_tensor = tf.convert_to_tensor(displacements)
        state_tensor = tf.expand_dims(state_tensor, 0)

        return state_tensor

    def update_inputs(self):
        t = time()

        self.sock.sendall(position_bytes)

        recv_bytes = self.sock.recv(17)
        self.current_position = np.frombuffer(recv_bytes, dtype=np.float32, count=3, offset=1)
        self.current_heading = np.frombuffer(recv_bytes, dtype=np.float32, count=1, offset=13) / 360

        self.one_sec_q.put((t, self.current_position))

        sleep(0.2)


if __name__ == '__main__':
    state_collector = InputCollector()

    get_keys()

    heading_array = np.zeros((3,), dtype=np.float32)
    heading_values = [328, 148]
    num_headings = 2
    n_mod = 0

    while True:
        # # state_collector.sock.sendall(teleport_bytes)
        #
        # headings = np.array([148, 328], dtype=np.float32)
        # heading_array = np.array([0, 0, 0], dtype=np.float32)
        # # heading_array[2] = rng(0, 360, 1).astype(np.float32)
        # # heading_array[2] = np.random.choice([148, 328]).astype(np.float32)
        # heading_array[2] = headings[n_mod]
        # n_mod = (n_mod + 1) % num_headings
        # heading_bytes = bytes([21, 0, 0, 0]) + heading_array.tobytes() + bytes([0])
        #
        # state_collector.sock.sendall(heading_bytes)
        # sleep(1)  # allow teleport to finish
        #
        # episode_reward = 0
        #
        # state_collector.current_position = starting_position.copy()
        # state_collector.one_sec_time = time()
        # state_collector.one_sec_position = starting_position.copy()
        # state_collector.five_sec_time = time()
        # state_collector.five_sec_position = starting_position.copy()
        # state_collector.ten_sec_time = time()
        # state_collector.ten_sec_position = starting_position.copy()
        #
        # state_collector.one_sec_q = SimpleQueue()
        # state_collector.five_sec_q = SimpleQueue()
        # state_collector.ten_sec_q = SimpleQueue()
        #
        # running_step_dist = 5
        # prev_pos = starting_position.copy()
        heading_values = heading_values[::-1]

        with tf.GradientTape() as tape:
            state_collector.sock.sendall(reinforcement_resumed_bytes)  # set control normals to 0
            state_collector.sock.sendall(teleport_bytes)
            heading_array[2] = heading_values[0]
            heading_bytes = bytes([21, 0, 0, 0]) + heading_array.tobytes() + bytes([0])
            state_collector.sock.sendall(heading_bytes)
            sleep(1)  # allow teleport to finish
            prev_pos = starting_position.copy()
            previous_distance = 0
            state_collector.update_inputs()
            print('sub_episode 1')
            for timestep in range(0, sub_episode_timesteps):
                state = state_collector.return_inputs()

                action_probs, critic_value = model(state)
                critic_value_history.append(critic_value[0, 0])
                # critic_value_history.append(tf.constant(2.0, dtype=tf.float32))

                action = np.random.choice(num_actions, p=np.squeeze(action_probs))

                keys = get_keys()
                # if 'J' in keys:
                #     action = 1
                # elif 'L' in keys:
                #     action = 2
                if 'I' in keys:
                    action = 0
                elif 'K' in keys:
                    action = 1

                action_probs_history.append(tf.math.log(action_probs[0, action]))

                # if timestep % 10 == 0:
                # if timestep == 1:
                if True:
                    print('state {}, action_probs {}'.format(state.numpy()[0], action_probs.numpy()[0]))

                set_controls(state_collector.sock, action)
                # sleep(0.25)  # give action time to take affect

                state_collector.update_inputs()

                current_distance = dist(state_collector.current_position, starting_position)
                step_distance = dist(state_collector.current_position, prev_pos)

                reward = (current_distance - previous_distance)
                # reward = step_distance
                # reward = 1.0
                rewards_history.append(reward)

                previous_distance = current_distance.copy()

                # running_step_dist = dist(state_collector.current_position, prev_pos) + 0.9 * running_step_dist

                prev_pos = state_collector.current_position.copy()

                # if running_step_dist < 3.0:
                #     print('running_step_dist < 1.0, ending at step {}'.format(timestep))
                #     break

                # if current_distance > 5:
                #     break

                # print(start_time - one_sec_time, one_sec_displacement, reward)
                # print(time() - start_time)

                # if sum(abs(state.numpy()[0, 4:])) < 0.6 and timestep > 25:
                #     print('Small displacements: {}'.format(state.numpy()[0]))
                #     print('Ending episode at step {}'.format(timestep))
                #     break

                if '5' in keys:
                    break

            state_collector.sock.sendall(reinforcement_resumed_bytes)  # set control normals to 0
            state_collector.sock.sendall(teleport_bytes)
            heading_array[2] = heading_values[1]
            heading_bytes = bytes([21, 0, 0, 0]) + heading_array.tobytes() + bytes([0])
            state_collector.sock.sendall(heading_bytes)
            sleep(1)  # allow teleport to finish
            prev_pos = starting_position.copy()
            previous_distance = 0
            state_collector.update_inputs()
            print('sub_episode 2')
            for timestep in range(0, sub_episode_timesteps):
                state = state_collector.return_inputs()

                action_probs, critic_value = model(state)
                critic_value_history.append(critic_value[0, 0])
                # critic_value_history.append(tf.constant(2.0, dtype=tf.float32))

                action = np.random.choice(num_actions, p=np.squeeze(action_probs))

                keys = get_keys()
                # if 'J' in keys:
                #     action = 1
                # elif 'L' in keys:
                #     action = 2
                if 'I' in keys:
                    action = 0
                elif 'K' in keys:
                    action = 1

                action_probs_history.append(tf.math.log(action_probs[0, action]))

                # if timestep % 10 == 0:
                # if timestep == 1:
                if True:
                    print('state {}, action_probs {}'.format(state.numpy()[0], action_probs.numpy()[0]))

                set_controls(state_collector.sock, action)
                # sleep(0.25)  # give action time to take affect

                state_collector.update_inputs()

                current_distance = dist(state_collector.current_position, starting_position)
                step_distance = dist(state_collector.current_position, prev_pos)

                reward = (current_distance - previous_distance)
                # reward = step_distance
                # reward = 1.0
                rewards_history.append(reward)

                previous_distance = current_distance.copy()
                prev_pos = state_collector.current_position.copy()

                if '5' in keys:
                    break

            # running_reward = 0.05 * current_distance + (1 - 0.05) * running_reward

            returns = []
            discounted_sum = 0
            for r in rewards_history[::-1]:
                discounted_sum = r + gamma * discounted_sum
                returns.insert(0, discounted_sum)

            print('rewards_history:')
            print(rewards_history)

            print('discounted returns:')
            print(returns)

            # returns = np.array(returns)
            # returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
            # returns = returns.tolist()
            returns = (np.array(returns) / 50.0).tolist()

            # print('normalised discounted returns')
            # print(returns)

            history = zip(action_probs_history, critic_value_history, returns)
            actor_losses = []
            critic_losses = []
            for log_prob, value, ret in history:
                print('prediction: {: 5.4}, actual {: 5.4}'.format(value, ret))
                diff = ret - value
                actor_losses.append(-log_prob * diff)
                critic_losses.append(huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0)))
                # print('predicted: {: 5.3}, actual: {: 5.3}'.format(value, ret))

            # print('critic: {: 5.4}, regularization: {: 5.3}'.format(sum(critic_losses).numpy(), sum(model.losses).numpy()))
            print('critic {: 5.4}, actor {: 5.4}'.format(sum(critic_losses).numpy(), sum(actor_losses).numpy()))

            loss_value = sum(actor_losses) + sum(critic_losses)
            # loss_value = sum(critic_losses) + sum(model.losses)
            grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            action_probs_history.clear()
            critic_value_history.clear()
            rewards_history.clear()

        episode_count += 1
        if episode_count % 1 == 0:
            print('End of episode {}'.format(episode_count))

        keys = get_keys()
        if '5' in keys:
            print('Quitting')
            break
        elif 'X' in keys:
            print('Saving model and quitting')
            model.save(model_file_name)
            break

    state_collector.sock.sendall(reinforcement_paused_bytes)
    sleep(1)  # wait for data to be sent
    state_collector.sock.close()
    # sock = None
    # recv_thread.join()

