from collections import deque

import keras
import numpy as np
import cv2
from keras import layers

# File can't be called 'numbers.py' because Python gets confused about imports

gamma = 0.63
epsilon_max = 0.1
max_steps_per_episode = 100

random_eval_action = None

env_name = 'Numbers'

sparse_reward_sequences_per_episode = 5
max_return = sparse_reward_sequences_per_episode * np.float32(1.0)


class Env:
    def __init__(self, eval_mode=False, progress_bar=False, num_actions=2, sparsity=1, action_delay=0):
        if num_actions < 1 or num_actions > 10:
            raise ValueError('num_actions must be between 1 and 10 inclusive')

        if sparsity < 1:
            raise ValueError('Minimum sparsity value is 1')

        if action_delay < 0:
            raise ValueError('Minimum action_delay value is 0')

        self.name = env_name
        self.input_shape = (16, 16)
        self.num_actions = num_actions

        self.timestep_dtype = np.dtype(
            [('observation', np.uint8, self.input_shape), ('action', np.int32), ('reward', np.float32)])
        self.observation_sequence_length = action_delay + 3
        self.eval_mode = eval_mode
        self.current_observation = None
        self.sparsity = sparsity
        self.progress_bar = progress_bar
        # self.episode_length = sparsity * sparse_reward_sequences_per_episode + action_delay
        self.episode_length = max_steps_per_episode
        # self.max_return = max_return
        self.max_return = self.episode_length

        if progress_bar and sparsity > 16:
            raise ValueError('Observation not wide enough for progress bar')

        self.step_count = 0
        self.action_delay = action_delay
        self.previous_numbers = deque(maxlen=action_delay + 1)
        self.correct_count = np.float32(0)
        self.images = np.zeros((num_actions,) + self.input_shape, dtype=np.uint8)

        for i, image in enumerate(self.images):
            cv2.putText(image, str(i), (2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,), 1)

    def reset(self):
        self.step_count = 0
        self.correct_count = np.float32(0)
        self.previous_numbers.clear()

        self.images[:, 0, 1:] = 0

        random_number = np.random.randint(self.num_actions)

        for _ in range(self.action_delay + 1):
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

        correct_action = action == delayed_number

        if self.step_count > self.action_delay:
            self.correct_count += np.float32(correct_action)

        random_number = np.random.randint(self.num_actions)

        self.previous_numbers.append(random_number)
        observation = self.images[random_number]

        self.current_observation = observation

        # never actually get gameover in this env
        #terminated = self.step_count == self.episode_length

        reward = np.float32(0)
        if self.step_count >= (self.action_delay + self.sparsity):
            if (self.step_count - self.action_delay) % self.sparsity == 0:
                reward = self.correct_count / self.sparsity
                self.correct_count = np.float32(0)
                self.images[:, 0, :self.sparsity] = 0

        if self.eval_mode:
            print(f'received action: {action}, delayed_number {delayed_number}, random_number: {random_number}, '
                  f'correct_action: {correct_action}, 'f'reward: {reward}')

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

    def close(self):
        cv2.destroyAllWindows()


def test_env():
    action_ords = list(map(ord, map(str, range(10))))
    env = Env(eval_mode=True)
    env.render(mode='namedWindow')
    action = 0
    terminated = False
    observation = env.reset()
    while True:
        if terminated:
            observation = env.reset()

        env.render(mode='human')

        cv2.imshow('Observation', observation)
        key = cv2.waitKey(0)

        if key == ord('q'):
            cv2.destroyAllWindows()
            break
        else:
            try:
                action = action_ords.index(key)
            except ValueError:
                pass

        observation, reward, terminated = env.step(action)


if __name__ == '__main__':
    test_env()
