from collections import deque

import cv2
import numpy as np
import gymnasium as gym
from keras.src.layers import Rescaling, Concatenate
from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.networks.q_network import QNetwork
from tf_agents.networks.q_rnn_network import QRnnNetwork
from tf_agents.specs import BoundedArraySpec
from tf_agents.trajectories.time_step import StepType, TimeStep

from common import resize

env_name = 'Breakout'
gamma = 0.99
max_return = 10_000  # Don't actually know what the max score is.

# Random policy's average return seems to be about 1.2


# This gets automatically turned into a TFPyEnvironment which in turn wraps the PyEnvironment in a BatchedPyEnvironment
# object. TFPyEnvironment overrides __getattr__ so that we can access attributes on the BatchedPyEnvironment object but
# BatchedPyEnvironment *does not* override __getattr__ so we can't access custom attributes on the PyEnvironment object.
class Env(PyEnvironment):
    """Provides Breakout as a PyEnvironment."""

    def __init__(self, discount=0.99, downscale=True, stack=True):
        super().__init__(handle_auto_reset=True)
        self.random_eval_action = 1
        self.downscale = downscale
        self.env = gym.make('BreakoutNoFrameskip-v4')
        self._action_spec = BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=3, name='action')

        if stack:
            self._observation_spec = BoundedArraySpec(shape=(84, 84, 4), dtype=np.float32, minimum=0, maximum=1.0,
                                                      name='observation')
        elif downscale:
            self._observation_spec = BoundedArraySpec(shape=(84, 84, 1), dtype=np.uint8, minimum=0, maximum=255,
                                                      name='observation')
        else:
            self._observation_spec = BoundedArraySpec(shape=(210, 160, 3), dtype=np.uint8, minimum=0, maximum=255,
                                                      name='observation')

        self.stack = stack
        if stack:
            self.frame_deque = deque(maxlen=4)

        self._discount = np.float32(discount)
        self.current_human_frame = None

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        frame, info = self.env.reset()

        frame = frame.astype(np.float32) / 255

        self.current_human_frame = frame

        if self.downscale:
            frame = resize(frame, height=84, width=84)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = np.expand_dims(frame, axis=-1)

        if self.stack:
            for _ in range(3):
                self.frame_deque.append(np.zeros((84, 84, 1), dtype=np.float32))

            self.frame_deque.append(frame)
            observation = np.concatenate(self.frame_deque, axis=2)
        else:
            observation = frame

        return TimeStep(StepType.FIRST, reward=np.asarray(0, dtype=np.float32), discount=self._discount,
                        observation=observation)

    def _step(self, action):
        frame, reward, terminated, truncated, info = self.env.step(action)

        frame = frame.astype(np.float32) / 255

        self.current_human_frame = frame

        if self.downscale:
            frame = resize(frame, height=84, width=84)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = np.expand_dims(frame, axis=-1)

        if self.stack:
            self.frame_deque.append(frame)
            observation = np.concatenate(self.frame_deque, axis=2)
        else:
            observation = frame

        if terminated:
            step_type = StepType.LAST
        else:
            step_type = StepType.MID

        return TimeStep(step_type, reward=np.asarray(reward, dtype=np.float32), discount=self._discount,
                        observation=observation)

    def create_q_net(self, recurrent=True):
        if recurrent:
            return QRnnNetwork(self.observation_spec(), self.action_spec(), lstm_size=(128,),
                               input_fc_layer_params=(256,), output_fc_layer_params=(256,),
                               # preprocessing_layers=Rescaling(1. / 255),
                               conv_layer_params=[(32, 8, 4), (64, 4, 2), (64, 3, 1)])
        else:
            return QNetwork(self.observation_spec(), self.action_spec(), fc_layer_params=(512,),
                            # preprocessing_layers=Rescaling(1. / 255),
                            conv_layer_params=[(32, 8, 4), (64, 4, 2), (64, 3, 1)])

    def close(self):
        cv2.destroyAllWindows()
        self.env.close()

    def render(self, mode='rgb_array'):
        if self.current_human_frame is None:
            self.current_human_frame = np.zeros((210, 160, 3), dtype=np.uint8)

        frame = cv2.cvtColor(self.current_human_frame, cv2.COLOR_BGR2RGB)
        frame = resize(frame, height=500)
        if mode == 'human':
            cv2.imshow('Breakout', frame)
            if cv2.waitKey(1) == ord('q'):
                cv2.destroyAllWindows()
                self.current_human_frame = None
        else:
            return frame


def test_env():
    env = Env()
    print(env.observation_spec())
    action = 0
    timestep = env.reset()
    while True:
        if timestep.step_type == StepType.LAST:
            timestep = env.reset()

        env.render(mode='human')

        if timestep.observation.shape[2] > 3:
            observation = np.hstack(np.split(timestep.observation, 4, axis=2))
        else:
            observation = timestep.observation

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

        timestep = env.step(action)


if __name__ == '__main__':
    test_env()
