from math import floor

import keras
import numpy as np
import cv2
from keras import layers

env_name = 'secret_sequence'
gamma = 0.99
epsilon_max = 1.0
max_steps_per_episode = 1_000

chars = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz+/='
secret_sequence = [1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0,
                   0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1]


class Env:
    def __init__(self, eval_mode=False, num_actions=2, depth=6):
        self.max_return = floor(max_steps_per_episode / depth)
        self.name = 'secret_sequence'
        self.timestep_dtype = np.dtype(
            [('observation', np.uint8, (16, 16)), ('action', np.int32), ('reward', np.float32)])

        self.correct_zeroth_action = False
        self.eval_mode = eval_mode
        self.depth = depth
        self.sequence_position = 0
        self.episode_length = depth

        self.num_actions = num_actions

        self.step_count = 0
        self.correct_sequence = True
        self.images = np.zeros((depth + 1,) + (16, 16), dtype=np.float32)
        self.current_observation = None

        for char, image in zip(chars, self.images):
            cv2.putText(image, char, (2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (1.0,), 1)

        self.secret_sequence = secret_sequence[:depth + 1]

        # self.secret_sequence[0] = 1

        assert len(self.secret_sequence) == depth + 1

        if num_actions != 2:
            raise NotImplementedError('Only 2 actions supported currently')

    def reset(self):
        self.step_count = 0
        self.correct_sequence = True

        observation = self.images[0]
        self.current_observation = observation

        if self.eval_mode:
            print(f'New episode, next secret number: {self.secret_sequence[0]}')

        return observation

    def step(self, action):
        self.sequence_position = self.step_count % self.depth

        correct_action = action == self.secret_sequence[self.sequence_position]

        if self.sequence_position == 0:
            self.correct_zeroth_action = correct_action

        self.correct_sequence = self.correct_sequence and correct_action

        observation = self.images[(self.step_count + 1) % self.depth]
        self.current_observation = observation

        reward = np.float32(0.0)
        if self.sequence_position == self.depth - 1:
            if self.correct_sequence:
                reward = np.float32(1.0)
            elif self.correct_zeroth_action:
                reward = np.float32(0.0)
            else:
                reward = np.float32(0.5)

        self.step_count += 1

        if self.eval_mode:
            print(f'received action: {action}, secret number was: {self.secret_sequence[self.sequence_position]}, '
                  f'correct_action: {correct_action}, 'f'reward: {reward}, correct_sequence: {self.correct_sequence}, '
                  f'next secret number: {self.secret_sequence[self.step_count % self.depth]}')

        if self.sequence_position == self.depth - 1:
            self.correct_sequence = True

        return observation, reward, False

    def render(self, mode='rgb_array'):
        cv2.imshow('OpenCV', self.current_observation)
        return cv2.waitKey(1)

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
