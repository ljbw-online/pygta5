from collections import deque

import keras
import numpy as np
import cv2
from keras import layers

# File can't be called 'numbers.py' because Python gets confused about imports

env_name = 'numbers'
gamma = 0.56
epsilon_max = 0.1

random_eval_action = None

action_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

sparse_reward_sequences_per_episode = 5
max_return = sparse_reward_sequences_per_episode * np.float32(1.0)


class Env:
    def __init__(self, eval_mode=False, progress_bar=False, num_actions=2, sparsity=1, delay=0):
        if num_actions < 1 or num_actions > 10:
            raise ValueError('num_actions must be between 1 and 10 inclusive')

        if sparsity < 1:
            raise ValueError('Minimum sparsity value is 1')

        if delay < 0:
            raise ValueError('Minimum action_delay value is 0')

        self.name = env_name
        # obs shape corresponds to what will go into replay buffer
        self.timestep_dtype = np.dtype(
            [('observation', np.uint8, (84, 84)), ('action', np.int32), ('reward', np.float32)])
        self.input_shape = self.timestep_dtype['observation'].shape
        self.num_actions = num_actions
        self.max_steps_per_episode = 100
        self.evaluation_epsilon = 0.0

        # self.observation_sequence_length = action_delay + 3
        self.eval_mode = eval_mode
        self.current_observation = None
        self.sparsity = sparsity
        self.progress_bar = progress_bar

        if progress_bar and sparsity > 16:
            raise ValueError('Observation not wide enough for progress bar')

        self.step_count = 0
        self.delay = delay
        self.previous_numbers = deque(maxlen=delay + 1)
        self.correct_count = np.float32(0)
        self.images = np.zeros((num_actions,) + self.input_shape, dtype=np.uint8)

        for i, image in enumerate(self.images):
            cv2.putText(image, str(i), (2, 76), cv2.FONT_HERSHEY_SIMPLEX, 3.2, (255,), 1)

    def reset(self):
        self.step_count = 0
        self.correct_count = np.float32(0)
        self.previous_numbers.clear()

        self.images[:, 0, 1:] = 0

        random_number = np.random.randint(self.num_actions)

        for _ in range(self.delay + 1):
            self.previous_numbers.append(random_number)

        observation = self.images[random_number]
        self.current_observation = observation

        if self.eval_mode:
            print(f'New episode,   random_number: {random_number}')

        return observation

    def step(self, action):
        if self.progress_bar:
            self.images[:, 0, self.step_count % self.sparsity] = 1

        self.step_count += 1

        delayed_number = self.previous_numbers.popleft()

        action_was_correct = action == delayed_number

        if self.step_count > self.delay:
            self.correct_count += np.float32(action_was_correct)

        random_number = np.random.randint(self.num_actions)

        self.previous_numbers.append(random_number)
        observation = self.images[random_number]

        self.current_observation = observation

        reward = np.float32(0)
        if self.step_count >= (self.delay + self.sparsity):
            if (self.step_count - self.delay) % self.sparsity == 0:
                reward = self.correct_count / self.sparsity
                self.correct_count = np.float32(0)
                self.images[:, 0, :self.sparsity] = 0

        if self.eval_mode:
            print(f'received action: {action}, delayed_number {delayed_number}, random_number: {random_number}, '
                  f'correct_action: {action_was_correct}, 'f'reward: {reward}')

        return observation, reward, False

    def render(self, mode=None):
        if mode == 'rgb_array':
            return self.current_observation
        elif mode == 'namedWindow':
            cv2.namedWindow('Numbers')
        else:
            frame = self.current_observation
            cv2.imshow(env_name, frame)
            cv2.waitKey(1)

    def create_q_net(self):
        inputs = layers.Input(shape=(16, 16))
        flatten = layers.Flatten()(inputs)
        dense = layers.Dense(128, activation='relu')(flatten)
        q_values = layers.Dense(self.num_actions, activation='linear')(dense)
        return keras.Model(inputs=inputs, outputs=q_values)

    def pause(self):
        return
    
    def close(self):
        cv2.destroyAllWindows()


def test_env():
    action_ords = list(map(ord, map(str, range(10))))
    env = Env(eval_mode=True)
    window_name = 'observation'
    cv2.namedWindow(window_name)
    key_ord = 0
    action = 0
    terminated = False
    observation = env.reset()
    while key_ord != ord('q'):
        if terminated:
            observation = env.reset()

        cv2.imshow(window_name, observation)
        key_ord = cv2.waitKey(0)

        try:
            action = action_ords.index(key_ord)
        except ValueError:
            pass

        observation_next, reward, terminated = env.step(action)

        # print(reward)
        # cv2.imshow('next', observation_next)

        observation = observation_next

    env.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    test_env()
