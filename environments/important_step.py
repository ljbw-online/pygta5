from collections import deque

import numpy as np
import cv2
from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.networks.q_network import QNetwork
from tf_agents.networks.q_rnn_network import QRnnNetwork
from tf_agents.specs import BoundedArraySpec
from tf_agents.trajectories.time_step import StepType, TimeStep

gamma = 1.0
random_eval_action = None

env_name = 'Numbers'

sparse_reward_sequences_per_episode = 1
max_return = sparse_reward_sequences_per_episode * np.float32(1.0)


# Work in progress
class Env(PyEnvironment):
    def __init__(self, discount=0.99, eval_mode=False, num_actions=2, sparsity=2):

        super().__init__(handle_auto_reset=True)

        if sparsity < 1:
            raise ValueError('Minimum sparsity value is 1')

        self._discount = np.float32(discount)
        self.eval_mode = eval_mode
        self.sparsity = sparsity
        self.episode_length = sparsity * sparse_reward_sequences_per_episode * 2

        self.num_actions = num_actions
        self._action_spec = BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=num_actions - 1,
                                             name='action')
        self._observation_spec = BoundedArraySpec(shape=(16, 16, 1), dtype=np.float32, minimum=0, maximum=1.0,
                                                  name='observation')

        self.step_count = 0

        self.important_step_action = None
        self.images = np.zeros((num_actions + 1,) + self._observation_spec.shape, dtype=self._observation_spec.dtype)

        for i, image in enumerate(self.images):
            cv2.putText(image, str(i), (2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (1.0,), 1)

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self.step_count = 0
        self.important_step_action = None

        observation = self.images[self.num_actions]

        if self.eval_mode:
            print(f'New episode')

        return TimeStep(StepType.FIRST, reward=np.asarray(0, dtype=np.float32), discount=self._discount,
                        observation=observation)

    def _step(self, action):
        self.step_count += 1

        reward = np.float32(0)
        observation = self.images[self.num_actions]

        if self.step_count == self.sparsity:
            self.important_step_action = np.random.randint(self.num_actions)
            observation = self.images[self.important_step_action]

        if self.step_count == self.sparsity + 1:
            self.important_step_correct = action == self.important_step_action

        if self.step_count == self.episode_length:
            step_type = StepType.LAST
            reward = np.float32(self.important_step_correct)
        else:
            step_type = StepType.MID

        if self.eval_mode:
            print(f'received action: {action}, important_step_action: {self.important_step_action}, '
                  f'reward: {reward}')

        return TimeStep(step_type, reward=np.asarray(reward, dtype=np.float32), discount=self._discount,
                        observation=observation)

    def render(self, mode='rgb_array'):
        if mode == 'human':
            frame = self._current_time_step.observation
            cv2.imshow(env_name, frame)
        elif mode == 'namedWindow':
            cv2.namedWindow('Numbers')
        else:
            return self._current_time_step.observation

    def create_q_net(self, recurrent=False):
        assert not recurrent
        return QRnnNetwork(self.observation_spec(), self.action_spec(), lstm_size=(128,))

    def close(self):
        cv2.destroyAllWindows()


def test_env():
    action_ords = list(map(ord, map(str, range(10))))
    env = Env(eval_mode=True)
    env.render(mode='namedWindow')
    action = 0
    timestep = env.reset()
    while True:
        if timestep.step_type == StepType.LAST:
            timestep = env.reset()

        env.render(mode='human')

        # cv2.imshow('Observation', timestep.observation)
        key = cv2.waitKey(0)

        if key == ord('q'):
            cv2.destroyAllWindows()
            break
        else:
            try:
                action = action_ords.index(key)
            except ValueError:
                pass

        timestep = env.step(action)


if __name__ == '__main__':
    test_env()
