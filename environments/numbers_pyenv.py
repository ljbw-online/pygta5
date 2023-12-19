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
    """A PyEnvironment in which observations consist of a digit depicted in a 16 by 16 black-and-white image.

    This environment provides an intermediate point between CartPole, which can be solved in seconds, and Breakout,
    which can take days to solve. It's very simple but has high-dimensional (i.e. visual) observations.

    The CartPole environment can be quickly solved by several different forms or reinforcement learning. For example,
    Deep Q-Learning with a starting epsilon value of 0.1 and a target update period of 1 will quickly solve CartPole.
    This makes it useful for basic debugging but not useful for verifying that an RL script can solve reasonably
    difficult environments such as Breakout. This environment was created for the purpose of performing such
    verification.

    In the easiest version of this environment the agent must choose the action that corresponds to the digit depicted
    in the observation. If the agent sees a "2" in the current observation then it must choose action 2 as its next
    action. If it does then it will receive a reward value of 1.0, otherwise it will receive 0.0.

    The environment's difficulty can be greatly increased with the constructor's keyword arguments.
    """
    def __init__(self, discount=0.99, eval_mode=False, progress_bar=False, num_actions=2, sparsity=1, action_delay=0):
        """
        Args:
            discount: (Optional) The value assigned to the discount attribute of the returned TimeSteps.

            eval_mode: (Optional) Prints information about each timestep, for debugging purposes.

            progress_bar: (Optional) If True then a progress bar is drawn across the top of the observations. This
                indicates to the agent how close it is to receiving a possibly-nonzero reward. The frequency with which
                nonzero reward values are returned is determined by the sparsity parameter.

            num_actions: (Optional) Specifies the number of valid actions and therefore also the number of digits that
                can be depicted.

            sparsity: (Optional) Number of timesteps between each non-zero reward value. A sparsity value of 1
                corresponds to getting a possibly-nonzero reward at every timestep. If sparsity is N then every N steps
                a reward will be returned which is equal to (correct_count / N) where correct_count is number of correct
                answers the agent gave in the preceding N steps. This emulates the Atari environments, in which the
                agent must associate current actions with possible reward values many steps into the future.

            action_delay: (Optional) Number of timesteps by which the images are offset from the actions expected of the
                agent. If action_delay is 0 then the agent must choose the action depicted in the current observation as
                its next action. If action_delay is 1 then the agent must choose the action number depicted in the
                *previous* observation as its next action. Hence if action_delay == N then the agent must have a way of
                knowing what number was depicted N steps ago. This provides a simple way of verifying that a recurrent
                Q-network is being trained as expected. Note that the preceding description is only true if
                sparsity == 1. If sparsity == M then the agent must still choose the action number depicted N steps ago
                but will only receive a nonzero reward every M steps.

        Raises:
            ValueError: If num_actions is not between 1 and 10 inclusive.
            ValueError: If sparsity is less than 1.
            ValueError: If action_delay is less than 0.
        """

        super().__init__(handle_auto_reset=True)

        if num_actions < 1 or num_actions > 10:
            raise ValueError('num_actions must be between 1 and 10 inclusive')

        if sparsity < 1:
            raise ValueError('Minimum sparsity value is 1')

        if action_delay < 0:
            raise ValueError('Minimum action_delay value is 0')

        self.name = env_name
        self.max_return = max_return
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
            cv2.imshow(env_name, frame)
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
