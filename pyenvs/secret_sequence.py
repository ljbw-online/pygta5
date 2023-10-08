import numpy as np
import cv2
from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.networks.q_network import QNetwork
from tf_agents.specs import BoundedArraySpec
from tf_agents.trajectories.time_step import StepType, TimeStep

env_name = 'secret_sequence'
gamma = 0.99

max_return = np.float32(1.0)
chars = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz+/='
secret_sequence = [0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0,
                   0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1]


class Env(PyEnvironment):
    def __init__(self, discount=0.99, eval_mode=False, num_actions=2, depth=4):

        super().__init__(handle_auto_reset=True)

        self._discount = np.float32(discount)
        self.eval_mode = eval_mode
        self.episode_length = depth

        self.num_actions = num_actions
        self._action_spec = BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=num_actions - 1,
                                             name='action')
        self._observation_spec = BoundedArraySpec(shape=(16, 16, 1), dtype=np.float32, minimum=0, maximum=1.0,
                                                  name='observation')

        self.step_count = 0
        self.correct_sequence = True
        self.images = np.zeros((depth + 1,) + self._observation_spec.shape, dtype=self._observation_spec.dtype)

        for char, image in zip(chars, self.images):
            cv2.putText(image, char, (2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (1.0,), 1)

        self.secret_sequence = secret_sequence[:depth + 1]

        self.secret_sequence[0] = 1

        assert len(self.secret_sequence) == depth + 1

        if num_actions != 2:
            raise NotImplementedError('Only 2 actions supported currently')

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self.step_count = 0
        self.correct_sequence = True

        observation = self.images[0]

        if self.eval_mode:
            print(f'New episode, secret_number: {self.secret_sequence[self.step_count]}')

        return TimeStep(StepType.FIRST, reward=np.asarray(0, dtype=np.float32), discount=self._discount,
                        observation=observation)

    def _step(self, action):
        self.step_count += 1

        correct_action = action == self.secret_sequence[self.step_count - 1]

        self.correct_sequence = self.correct_sequence and correct_action

        observation = self.images[self.step_count]

        reward = np.float32(0)
        if self.step_count == self.episode_length:
            step_type = StepType.LAST
            if self.correct_sequence:
                reward = np.float32(self.correct_sequence)
            else:
                reward = np.float32(0.5)
        else:
            step_type = StepType.MID

        if self.eval_mode:

            print(f'received action: {action}, secret number: {self.secret_sequence[self.step_count - 1]}, '
                  f'correct_action: {correct_action}, 'f'reward: {reward}, correct_sequence: {self.correct_sequence}, '
                  f'next secret number: {self.secret_sequence[self.step_count]}')

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
            cv2.namedWindow(env_name)
        else:
            return self._current_time_step.observation

    def create_q_net(self, recurrent=False):
        assert not recurrent
        return QNetwork(self.observation_spec(), self.action_spec(), fc_layer_params=(256, 256))

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
