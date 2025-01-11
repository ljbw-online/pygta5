from collections import deque

import cv2
import numpy as np
import gymnasium as gym
import ale_py

from common import resize

# Random policy's average return seems to be about 1.2
# Deep Q-Learning was getting between 5 and 10 points per episode after running overnight
# Double Deep Q-Learning was getting just over 9 points per episode on average after 4.3 million iterations
# Dueling Architecture Double Deep Q-Learning with one fully connected layer was getting 8 points on average after
# 4.7 million iterations
# Dueling Architecture Double Deep Q-Learning with two 512-unit fully connected layers had regressed down to 6.5 points
# on average after 4.4 million iterations. By 30 million iterations it had not recovered.
# Same config as above, but with a learning rate of 1e-5, was scoring 10.1 points after 6.9 million iterations. It
# achieved a maximum average score of about 47 points after almost 50 million iterations. I stopped the training run
# because it was really struggling to keep the average score in the high 40s.
# Same as above, but with a replay buffer of 1000 episodes, was scoring 5 points after 4.5m iterations. Got a evaluation
# score of 22.5 after 10m iterations.

env_name = 'breakout'
gamma = 0.99
epsilon_max = 1.0
# max_steps_per_episode = 100_000
action_labels = ['nothing', 'start', '-->', '<--']

gym.register_envs(ale_py)

class Env:
    def __init__(self):
        self.name = env_name
        self.random_eval_action = 1
        self.max_return = None
        self.env = gym.make('ALE/Breakout-v5')
        self.num_actions = 4
        self.max_steps_per_episode = 100_000
        self.evaluation_epsilon = 0.05

        # if stack:
        #     input_channels = 4
        # else:
        #     input_channels = 1

        self.timestep_dtype = np.dtype([('observation', np.uint8, (84, 84)), ('action', np.int32),
                                        ('reward', np.float32)])
        # self.clip_rewards = clip_rewards

        # self.stack = stack
        # if stack:
        #     self.frame_deque = deque(maxlen=4)

        self.current_human_frame = None

    def reset(self):
        frame, info = self.env.reset()

        self.current_human_frame = frame

        # if self.stack:
        #     frame = resize(frame, height=84, width=84)
        #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #     frame = np.expand_dims(frame, axis=-1)
        #
        #     for _ in range(3):
        #         self.frame_deque.append(np.zeros((84, 84, 1), dtype=np.uint8))
        #
        #     self.frame_deque.append(frame)
        #     observation = np.concatenate(self.frame_deque, axis=2)
        # else:
        observation = frame

        return observation

    def step(self, action):
        frame, reward, terminated, truncated, info = self.env.step(action)

        self.current_human_frame = frame

        # if self.stack:
        #     frame = resize(frame, height=84, width=84)
        #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #     frame = np.expand_dims(frame, axis=-1)
        #     self.frame_deque.append(frame)
        #     observation = np.concatenate(self.frame_deque, axis=2)
        # else:
        observation = frame

        # if self.clip_rewards:
        #     reward = min(reward, 1.0)
        #     reward = np.float32(reward)

        # if terminated:
        #     reward = np.float32(-1)

        return observation, reward, terminated

    def pause(self):
        return

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
    key_ord = 0
    terminated = False
    observation = env.reset()
    while key_ord != ord('q'):
        if terminated:
            observation = env.reset()

        env.render(mode='human')

        cv2.imshow('Observation', observation)
        key_ord = cv2.waitKey(0)

        if observation.shape[2] > 3:
            observation = np.hstack(np.split(observation, 4, axis=2))

        if key_ord == ord('j'):
            action = 3
        elif key_ord == ord('k'):
            action = 0
        elif key_ord == ord('l'):
            action = 2
        elif key_ord == ord('i'):
            action = 1

        observation_next, reward, terminated = env.step(action)

        # A reward of 1.0 is given with the first observation in which a brick has disappeared.
        print(reward)

        observation = observation_next

    env.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    test_env()
