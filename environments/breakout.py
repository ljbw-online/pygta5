from collections import deque

import cv2
import numpy as np
import gymnasium as gym
import keras
from keras import layers

from common import resize

# Random policy's average return seems to be about 1.2
# Deep Q-Learning was getting between 5 and 10 points per episode after running overnight
# Double Deep Q-Learning was getting just over 9 points per episode on average after 4.3 million iterations
# Dueling Architecture Double Deep Q-Learning with one fully connected layer was getting 8 points on average after
# 4.7 million iterations
# Dueling Architecture Double Deep Q-Learning with two 512-unit fully connected layers had regressed down to 6.5 points
# on average after 4.4 million iterations. By 30 million iterations it had not recovered.
# Same config as above, but with a learning rate of 1e-5, was scoring 10.1 points after 6.9 million iterations.

gamma = 0.99
epsilon_max = 1.0
max_steps_per_episode = 100_000


class Env:
    def __init__(self, stack=True, clip_rewards=True):
        self.name = 'Breakout'
        self.random_eval_action = 1
        self.max_return = None
        self.env = gym.make('BreakoutNoFrameskip-v4')
        self.num_actions = 4
        self.input_shape = (84, 84, 4)

        self.timestep_dtype = np.dtype(
            [('observation', np.uint8, (84, 84, 4)), ('action', np.int32), ('reward', np.float32)])
        self.clip_rewards = clip_rewards

        self.stack = stack
        if stack:
            self.frame_deque = deque(maxlen=4)

        self.current_human_frame = None

    def reset(self):
        frame, info = self.env.reset()

        self.current_human_frame = frame

        frame = resize(frame, height=84, width=84)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = np.expand_dims(frame, axis=-1)

        if self.stack:
            for _ in range(3):
                self.frame_deque.append(np.zeros((84, 84, 1), dtype=np.uint8))

            self.frame_deque.append(frame)
            observation = np.concatenate(self.frame_deque, axis=2)
        else:
            observation = frame

        return observation

    def step(self, action):
        frame, reward, terminated, truncated, info = self.env.step(action)

        self.current_human_frame = frame

        frame = resize(frame, height=84, width=84)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = np.expand_dims(frame, axis=-1)

        if self.stack:
            self.frame_deque.append(frame)
            observation = np.concatenate(self.frame_deque, axis=2)
        else:
            observation = frame

        if self.clip_rewards:
            reward = min(reward, 1.0)
            reward = np.float32(reward)

        # if terminated:
        #     reward = np.float32(-1)

        return observation, reward, terminated

    def create_q_net(self):
        inputs = layers.Input(shape=(84, 84, 4))
        rescaling = layers.Rescaling(1. / 255)(inputs)

        conv1 = layers.Conv2D(32, 8, strides=4, activation='relu')(rescaling)
        conv2 = layers.Conv2D(64, 4, strides=2, activation='relu')(conv1)
        conv3 = layers.Conv2D(64, 3, strides=1, activation='relu')(conv2)

        flatten = layers.Flatten()(conv3)

        dense = layers.Dense(512, activation='relu')(flatten)
        q_values = layers.Dense(self.num_actions, activation='linear')(dense)

        return keras.Model(inputs=inputs, outputs=q_values)

    def close(self):
        cv2.destroyAllWindows()
        self.env.close()

    def render(self, mode=None):
        if self.current_human_frame is None:
            self.current_human_frame = np.zeros((210, 160, 3), dtype=np.uint8)

        frame = cv2.cvtColor(self.current_human_frame, cv2.COLOR_BGR2RGB)
        frame = resize(frame, height=500)
        if mode == 'rgb_array':
            return frame
        else:
            cv2.imshow('Breakout', frame)
            if cv2.waitKey(1) == ord('q'):
                cv2.destroyAllWindows()
                self.current_human_frame = None


def test_env():
    env = Env()
    action = 0
    reward = 0.0
    terminated = False
    observation = env.reset()
    while True:
        if terminated:
            observation = env.reset()
            reward = 0.0

        env.render(mode='human')
        print(reward)

        if observation.shape[2] > 3:
            observation = np.hstack(np.split(observation, 4, axis=2))

        cv2.imshow('Observation', observation)
        key = cv2.waitKey(0)

        if key == ord('q'):
            cv2.destroyAllWindows()
            env.close()
            break
        elif key == ord('j'):
            action = 3
        elif key == ord('k'):
            action = 0
        elif key == ord('l'):
            action = 2
        elif key == ord('i'):
            action = 1

        observation, reward, terminated = env.step(action)


if __name__ == '__main__':
    test_env()
