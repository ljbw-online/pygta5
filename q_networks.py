from collections import deque

import cv2
import numpy as np
from keras import layers, Model
import tensorflow as tf

from common import resize


def fully_connected(input_shape, num_actions):
    inputs = layers.Input(shape=input_shape)
    rescaling = layers.Rescaling(1. / 255)(inputs)
    flatten = layers.Flatten()(rescaling)
    dense = layers.Dense(128, activation='relu')(flatten)
    q_values = layers.Dense(num_actions, activation='linear')(dense)
    return Model(inputs=inputs, outputs=q_values)


def rescale_conv_dense(input_shape, num_actions):
    inputs = layers.Input(shape=input_shape)
    rescaling = layers.Rescaling(1. / 255)(inputs)

    conv1 = layers.Conv2D(32, 8, strides=4, activation='relu')(rescaling)
    conv2 = layers.Conv2D(64, 4, strides=2, activation='relu')(conv1)
    conv3 = layers.Conv2D(64, 3, strides=1, activation='relu')(conv2)

    flatten = layers.Flatten()(conv3)

    dense = layers.Dense(512, activation='relu')(flatten)
    q_values = layers.Dense(num_actions, activation='linear')(dense)

    return Model(inputs=inputs, outputs=q_values)


def dueling_architecture(input_shape, num_actions, testing=False):
    inputs = layers.Input(shape=input_shape)

    # if input_shape[0] < 64:
    #     resizing = layers.Resizing(64, 64)(inputs)
    #     rescaling = layers.Rescaling(1. / 255)(resizing)
    # else:
    rescaling = layers.Rescaling(1. / 255)(inputs)

    conv1 = layers.Conv2D(32, 8, strides=4, activation='relu')(rescaling)
    conv2 = layers.Conv2D(64, 4, strides=2, activation='relu')(conv1)
    conv3 = layers.Conv2D(64, 3, strides=1, activation='relu')(conv2)

    flatten = layers.Flatten()(conv3)

    value_dense = layers.Dense(512, activation='relu')(flatten)
    value = layers.Dense(1, activation='linear')(value_dense)  # Dense(N, ...) has shape (batch, N)

    advantage_dense = layers.Dense(512, activation='relu')(flatten)
    advantages = layers.Dense(num_actions, activation='linear')(advantage_dense)

    # AveragePooling1D expects (batch, steps, features)
    reshaped_advantages = layers.Reshape((num_actions, 1))(advantages)

    average_advantage = layers.AveragePooling1D(num_actions)(reshaped_advantages)
    average_advantage = layers.Reshape((1,))(average_advantage)  # AveragePooling1D outputs (batch, steps, features)

    value_minus_average_advantage = layers.Subtract()([value, average_advantage])

    repeated_difference = layers.RepeatVector(num_actions)(value_minus_average_advantage)  # (batch, num_actions, 1)
    repeated_difference = layers.Reshape((num_actions,))(repeated_difference)

    action_values = layers.Add()([repeated_difference, advantages])

    if testing:
        outputs = [value, advantages, action_values]
    else:
        outputs = action_values

    return Model(inputs=inputs, outputs=outputs)


def downsample_observation(observation):
    if observation.shape[0] > 84:
        observation = resize(observation, width=84, height=84)

    try:
        if observation.shape[2] == 3:
            return cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
    except IndexError:
        pass

    return observation


class QFunction:
    def __init__(self, obs_seq_len, env, model):
        self.obs_seq_len = obs_seq_len
        self.num_actions = env.num_actions
        self.obs_shape = env.timestep_dtype['observation'].shape
        self.obs_dtype = env.timestep_dtype['observation'].subdtype[0]
        self.obs_deque = deque(maxlen=obs_seq_len)
        self.model = model
        self.downsampler = downsample_observation
        self.obs_stored = 0
        self.clear()

    def __call__(self, obs):
        downsampled_obs = self.downsampler(obs)
        obs = np.expand_dims(downsampled_obs, 2)
        self.obs_deque.append(obs)
        self.obs_stored = min(self.obs_stored + 1, self.obs_seq_len)
        model_input = np.concatenate(self.obs_deque, axis=2)
        model_input = np.expand_dims(model_input, axis=0)  # add batch dimension
        model_input = tf.convert_to_tensor(model_input)

        if self.obs_stored < self.obs_seq_len:
            q_values = tf.zeros((self.num_actions,), dtype=tf.float32)
        else:
            q_values = self.model(model_input, training=False)[0]

        return q_values, downsampled_obs

    def clear(self):
        # for _ in range(self.obs_seq_len - 1):
        #    self.obs_deque.append(np.zeros(self.obs_shape + (1,), dtype=self.obs_dtype))
        self.obs_deque.clear()
        self.obs_stored = 0


def test_q_function():
    import os

    import cv2

    from common import base_dir
    from environments.numbers_env import Env, env_name
    from dqn import run_episode
    from replay_buffer import ReplayBuffer

    env = Env(sparsity=4, num_actions=4)

    save_dir = os.path.join(base_dir, env_name)
    replay_buffer_dir = os.path.join(save_dir, 'replay_buffer')

    try:
        os.makedirs(replay_buffer_dir)
    except FileExistsError:
        pass

    obs_seq_len = 4
    rb = ReplayBuffer(replay_buffer_dir, env, obs_seq_len=obs_seq_len)

    def placeholder_model(observation, training=False):
        return observation

    qf = QFunction(obs_seq_len, env, placeholder_model)

    episode, terminated, _, _ = run_episode(env, qf, 1.0)

    if terminated:
        episode['reward'][-1] = -1

    rb.add_episode(episode)

    for i in range(len(episode) - obs_seq_len):
        qf(episode['observation'][i])
        qf(episode['observation'][i + 1])
        qf(episode['observation'][i + 2])
        frame_stack, _ = qf(episode['observation'][i + 3])

        frame_stack = frame_stack.numpy()
        axis_moved = np.moveaxis(episode['observation'][i:i+4], 0, -1)

        assert np.array_equal(frame_stack, axis_moved)

        fs_h = np.hstack((frame_stack[:, :, 0], frame_stack[:, :, 1], frame_stack[:, :, 2], frame_stack[:, :, 3]))
        am_h = np.hstack((axis_moved[:, :, 0], axis_moved[:, :, 1], axis_moved[:, :, 2], axis_moved[:, :, 3]))

        cv2.imshow('mn_conc', fs_h)
        cv2.imshow('ma_conc', am_h)
        cv2.waitKey(0)

    cv2.destroyAllWindows()


def test_dueling_architecture():
    input_shape = (84, 84, 4)
    num_actions = 3
    rng = np.random.default_rng()

    random_input = rng.integers(0, 255, size=np.prod(input_shape), dtype=np.uint8).reshape((1,) + input_shape)

    model = dueling_architecture(input_shape, num_actions, testing=True)

    model.summary()

    test_output = model(random_input)

    test_value = test_output[0].numpy()
    test_advantages = test_output[1].numpy()
    test_action_values = test_output[2].numpy()

    expected_action_values = test_value + test_advantages - np.mean(test_advantages)

    print(test_output)
    print(expected_action_values)
    print(test_action_values)

    assert np.max(np.abs(expected_action_values - test_action_values)) < 1e-8  # array_equal isn't always True
    assert test_action_values.shape == (1, num_actions)
    print('Assertions passed.')


if __name__ == '__main__':
    test_q_function()
