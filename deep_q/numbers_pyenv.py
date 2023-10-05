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

sparse_reward_sequences_per_episode = 5
max_return = sparse_reward_sequences_per_episode * np.float32(1.0)


class Env(PyEnvironment):
    def __init__(self, discount=0.99, eval_mode=False, num_actions=2, sparsity=1, progress_bar=False,
                 action_delay=1):

        super().__init__(handle_auto_reset=True)

        if sparsity < 1:
            raise ValueError('Minimum sparsity value is 1')

        if action_delay < 0:
            raise ValueError('Minimum action_delay value is 0')

        self._discount = np.float32(discount)
        self.observation_sequence_length = action_delay + 3
        self.eval_mode = eval_mode
        self.sparsity = sparsity
        self.progress_bar = progress_bar
        self.episode_length = sparsity * sparse_reward_sequences_per_episode + action_delay

        self.num_actions = num_actions
        self._action_spec = BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=num_actions - 1,
                                             name='action')
        self._observation_spec = BoundedArraySpec(shape=(16, 16, 1), dtype=np.float32, minimum=0, maximum=1.0,
                                                  name='observation')

        if progress_bar and sparsity > self._observation_spec.shape[1]:
            raise ValueError('Observation not wide enough for progress bar')

        self.step_count = 0
        # self.prev_number = None
        self.action_delay = action_delay
        self.previous_numbers = deque(maxlen=action_delay + 1)
        self.correct_count = np.float32(0)
        self.images = np.zeros((num_actions,) + self._observation_spec.shape, dtype=self._observation_spec.dtype)

        for i, image in enumerate(self.images):
            cv2.putText(image, str(i), (2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (1.0,), 1)

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self.step_count = 0
        self.correct_count = np.float32(0)
        self.previous_numbers.clear()

        self.images[:, 0, 1:] = 0

        random_number = np.random.randint(self.num_actions)

        for _ in range(self.action_delay + 1):
            self.previous_numbers.append(random_number)

        observation = self.images[random_number]

        if self.eval_mode:
            print(f'New episode,   random_number: {random_number}')

        return TimeStep(StepType.FIRST, reward=np.asarray(0, dtype=np.float32), discount=self._discount,
                        observation=observation)

    def _step(self, action):
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

        if self.step_count == self.episode_length:
            step_type = StepType.LAST
        else:
            step_type = StepType.MID

        reward = np.float32(0)
        if self.step_count >= (self.action_delay + self.sparsity):
            if (self.step_count - self.action_delay) % self.sparsity == 0:
                reward = self.correct_count / self.sparsity
                self.correct_count = np.float32(0)
                self.images[:, 0, :self.sparsity] = 0

        if self.eval_mode:
            print(f'received action: {action}, delayed_number {delayed_number}, random_number: {random_number}, '
                  f'correct_action: {correct_action}, 'f'reward: {reward}')

        return TimeStep(step_type, reward=np.asarray(reward, dtype=np.float32), discount=self._discount,
                        observation=observation)

    def render(self, mode='rgb_array'):
        if mode == 'human':
            frame = self._current_time_step.observation
            # frame = resize(frame, height=500)
            cv2.imshow(env_name, frame)
            # if cv2.waitKey(1) == ord('q'):
            #     cv2.destroyAllWindows()
        elif mode == 'namedWindow':
            cv2.namedWindow('Numbers')
        else:
            return self._current_time_step.observation

    def create_q_net(self):
        # If conv_layer_params is None then this just applies a flattening layer.
        if self.action_delay == 0:
            return QNetwork(self.observation_spec(), self.action_spec(), fc_layer_params=(256, 256))
        else:
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
