from collections import deque
from time import sleep, time
import os
from pathlib import Path

import cv2
import numpy as np
import gymnasium

from common import uint8_to_float32

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from tensorflow import keras as ks

env_name = os.path.basename(__file__.removesuffix('.py'))

stack_len = 1
input_shape = (105, 80, 3)
num_actions = 4  # Used in training script


def create_q_model():
    inputs = ks.layers.Input(shape=input_shape)
    rescaled = ks.layers.Rescaling(1. / 255)(inputs)

    conv1_filters = 6

    conv1_reg = 1 / (32 * conv1_filters * 21 * 39)
    conv2_reg = 1 / (32 * 16 * 9 * 18)
    conv3_reg = 1 / (32 * 32 * 7 * 16)

    layer1 = ks.layers.Conv2D(conv1_filters, 8, strides=2, activation='relu',
                              activity_regularizer=ks.regularizers.l1(conv1_reg))(rescaled)

    layer2 = ks.layers.Conv2D(16, 4, strides=2, activation='relu',
                              activity_regularizer=ks.regularizers.l1(conv2_reg))(layer1)

    layer3 = ks.layers.Conv2D(32, 3, strides=2, activation='relu',
                              activity_regularizer=ks.regularizers.l1(conv3_reg))(layer2)

    layer4 = ks.layers.Flatten()(layer3)

    layer5 = ks.layers.Dense(64, activation='relu')(layer4)
    q_values = ks.layers.Dense(num_actions, activation=ks.activations.linear)(layer5)

    return ks.Model(inputs=inputs, outputs=q_values)


def visualise(q):
    wasd_im = np.zeros((360, 640), dtype=np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    pressed_colour = (170,)
    non_pressed_colour = (85,)
    letter_colour = (255,)

    while True:
        q_item = q.get(block=True)

        if q_item is None:
            cv2.destroyAllWindows()
            return
        else:
            frame_list, conv_stack, action, random_action, q_values, reward = q_item

        frames = frame_list

        frames = cv2.resize(frames, (640, 360), interpolation=cv2.INTER_NEAREST)
        frames = uint8_to_float32(frames)

        conv_list = []
        conv_stack_len = conv_stack.shape[2]
        for i in range(0, conv_stack_len, 2):
            conv_list.append(np.hstack((conv_stack[:, :, i], conv_stack[:, :, i + 1])))

        convs = np.vstack(conv_list)
        convs = cv2.resize(convs, (640, 180 * int(conv_stack_len / 2)), interpolation=cv2.INTER_NEAREST)
        convs = cv2.cvtColor(convs, cv2.COLOR_GRAY2RGB)

        frames_convs = np.vstack((frames, convs))

        cv2.imshow('INPUT & CONV1', frames_convs)

        q_values = np.array(q_values)

        a0 = q_values[0]
        a1 = q_values[1]
        a2 = q_values[2]
        a3 = q_values[3]

        if random_action:
            streamer_text_colour = (255,)
            ai_text_colour = (85,)
        else:
            streamer_text_colour = (85,)
            ai_text_colour = (255,)

        cv2.putText(wasd_im, 'Random', (25, 50), font, 1, streamer_text_colour, 2)
        cv2.putText(wasd_im, 'AI: {}'.format(action), (25, 100), font, 1, ai_text_colour, 2)
        cv2.putText(wasd_im, 'Reward: {:.2f}'.format(reward), (25, 200), font, 1, (255,), 2)
        cv2.putText(wasd_im, '{: .2f} {: .2f} {: .2f} {: .2f}'.format(a0, a1, a2, a3), (25, 150), font, 1, (255,), 2)

        cv2.imshow('WASD', wasd_im)
        cv2.waitKey(1)

        cv2.putText(wasd_im, 'Reward: {:.2f}'.format(reward), (25, 200), font, 1, (0,), 2)
        cv2.putText(wasd_im, 'AI: {}'.format(action), (25, 100), font, 1, (0,), 2)
        cv2.putText(wasd_im, '{: .2f} {: .2f} {: .2f} {: .2f}'.format(a0, a1, a2, a3), (25, 150), font, 1, (0,), 2)

def apply_correction(keys, action):
    corrected = False
    if 'J' in keys:
        corrected = True
        action = 3
    elif 'K' in keys:
        corrected = True
        action = 2

    return action, corrected

class Env:
    def __init__(self):
        self.env = gymnasium.make('BreakoutNoFrameskip-v4')

    def reset(self):
        state, info = self.env.reset()
        return cv2.resize(state, (80, 105), interpolation=cv2.INTER_NEAREST)

    def step(self, action, keys):
        state, reward, terminated, truncated, info = self.env.step(action)

        if terminated:
            reward = np.float32(-1)

        resized_state = cv2.resize(state, (80, 105), interpolation=cv2.INTER_NEAREST)
        return resized_state, reward, terminated

    def pause(self):
        return

    def resume(self):
        return

    def close(self):
        self.env.close()
