import numpy as np
from keras import layers, Model
import tensorflow as tf


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
        return Model(inputs=inputs, outputs=[value, advantages, action_values])
    else:
        return Model(inputs=inputs, outputs=action_values)


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
    test_dueling_architecture()
