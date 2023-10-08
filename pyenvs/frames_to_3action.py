import pickle
from collections import deque
from math import floor
from queue import Queue
from socket import socket
from time import sleep, time
import os
from pathlib import Path

import cv2
import numpy as np

from common import reinforcement_resumed_bytes, w_bytes, s_bytes, wa_bytes, wd_bytes, state_bytes, \
    reinforcement_paused_bytes, sa_bytes, sd_bytes, a_bytes, d_bytes, nk_bytes, uint8_to_float32
from windows import get_frame

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from tensorflow import keras as ks

# Doesn't actually implement PyEnvironment yet.

file_name = os.path.basename(__file__.removesuffix('.py'))
model_path = os.path.join(Path.home(), 'My Drive\\Models', file_name)
target_model_path = os.path.join(Path.home(), 'My Drive\\Models', file_name + 'target')
data_path = os.path.join(Path.home(), 'Documents\\Data', file_name)

stack_len = 4
input_shape = (stack_len, 90, 160)
num_actions = 3  # Used in training script

starting_position = np.array([1099, -265, 69], dtype=np.float32)
teleport_bytes = bytes([19, 0, 0, 0]) + starting_position.tobytes() + bytes([0])
headings = [147, 327]
num_headings = len(headings)


def heading_to_bytes(heading):
    return bytes([21, 0, 0, 0]) + np.array([0, 0, heading], dtype=np.float32).tobytes() + bytes([0])


def create_q_model():
    inputs = ks.layers.Input(shape=input_shape)
    rescaled = ks.layers.Rescaling(1. / 255)(inputs)

    conv1_filters = 6

    conv1_reg = 1 / (32 * conv1_filters * 21 * 39)
    conv2_reg = 1 / (32 * 16 * 9 * 18)
    conv3_reg = 1 / (32 * 32 * 7 * 16)

    # Based on the Deep-Q implementation for Breakout, on keras.io
    layer1 = ks.layers.Conv2D(conv1_filters, 8, strides=2, activation='relu', data_format='channels_first',
                              activity_regularizer=ks.regularizers.l1(conv1_reg))(rescaled)

    layer2 = ks.layers.Conv2D(16, 4, strides=2, activation='relu', data_format='channels_first',
                              activity_regularizer=ks.regularizers.l1(conv2_reg))(layer1)

    layer3 = ks.layers.Conv2D(32, 3, strides=2, activation='relu', data_format='channels_first',
                              activity_regularizer=ks.regularizers.l1(conv3_reg))(layer2)

    layer4 = ks.layers.Flatten()(layer3)

    layer5 = ks.layers.Dense(64, activation='relu')(layer4)
    action_probs = ks.layers.Dense(num_actions, activation=ks.activations.linear)(layer5)

    return ks.Model(inputs=inputs, outputs=[action_probs, layer1])


def apply_correction(keys, action):
    corrected = False
    # w <no-keys> s
    if 'I' in keys:
        corrected = True
        action = 0
    elif 'K' in keys:
        corrected = True
        action = 2
    # else: # Currently indistinguishable from me stopping my corrections to let the model drive.
    #     action = 8

    return action, corrected


def visualise(q):
    last_corrected_time = time()
    wasd_im = np.zeros((360, 640), dtype=np.uint8)
    frames_convs = np.zeros((720, 640), dtype=np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    pressed_colour = (170,)
    non_pressed_colour = (85,)
    letter_colour = (255,)

    # w wa wd s sa sd a d nk
    #                    w          nk          s
    square_top_lefts = [(410, 20), (410, 130), (410, 240)]
    # tl_grid =[(300, 20), (410, 20), (520, 20), (300, 130), (410, 130), (520, 130), (300, 240), (410, 240), (520, 240)]
    #                       w           nk           s
    square_bottom_rights = [(510, 120), (510, 230), (510, 340)]
    # br_grid=[(400, 120), (510, 120), (620, 120), (400, 230), (510, 230), (620, 230), (400, 340),(510, 340),(620, 340)]
    letter_bottom_lefts = [(440, 70), (440, 180), (440, 290)]
    number_bottom_lefts = [(430, 100), (430, 210), (430, 330)]
    # old_letter_bls = [(330, 90), (440, 90), (550, 90), (330, 200), (440, 200),(550,200),(330,310),(440,310),(550,310)]
    letters = ['W', 'NK', 'S']

    max_qs = deque(maxlen=500)

    while True:
        q_item = q.get(block=True)

        if q_item is None:
            cv2.destroyAllWindows()
            return
        else:
            frame_list, conv_stack, action, random_action, q_values, reward = q_item

        frame01 = np.hstack((frame_list[0], frame_list[1]))
        frame23 = np.hstack((frame_list[2], frame_list[3]))

        frames = np.vstack((frame01, frame23))
        frames = cv2.resize(frames, (640, 360), interpolation=cv2.INTER_NEAREST)
        frames = uint8_to_float32(frames)

        conv_list = []
        for i in range(0, len(conv_stack), 2):
            conv_list.append(np.hstack((conv_stack[i], conv_stack[i + 1])))

        convs = np.vstack(conv_list)
        convs = cv2.resize(convs, (640, 180 * int(len(conv_stack) / 2)), interpolation=cv2.INTER_NEAREST)

        frames_convs = np.vstack((frames, convs))

        cv2.imshow('INPUT & CONV1', frames_convs)

        max_q_value = float(max(q_values))
        max_qs.append(max_q_value)
        mean_max_q = np.mean(max_qs)
        q_values = np.array(q_values)

        wasd_zip = zip(square_top_lefts, square_bottom_rights, letter_bottom_lefts, letters, q_values,
                       number_bottom_lefts)

        for square_top_left, square_bottom_right, letter_bottom_left, letter, q_value, number_bottom_left in wasd_zip:
            q_value_float = q_value
            q_value = q_value / mean_max_q  # np.average takes an optional weights argument
            q_value = np.clip(q_value * 255, 0, 255).astype(np.uint8)
            cv2.rectangle(wasd_im, square_top_left, square_bottom_right, (int(q_value),), cv2.FILLED)
            cv2.putText(wasd_im, letter, letter_bottom_left, font, 1, (0,), 2)
            cv2.putText(wasd_im, '{:.2f}'.format(q_value_float), number_bottom_left, font, 1, (0,), 2)

        action_top_left = square_top_lefts[action]
        action_bottom_right = square_bottom_rights[action]
        action_top_left_2 = (action_top_left[0] + 2, action_top_left[1] + 2)
        action_bottom_right_2 = (action_bottom_right[0] - 2, action_bottom_right[1] - 2)
        cv2.rectangle(wasd_im, action_top_left, action_bottom_right, (255,), 1)
        cv2.rectangle(wasd_im, action_top_left_2, action_bottom_right_2, (0,), 2)

        if random_action:
            streamer_text_colour = (255,)
            ai_text_colour = (85,)
        else:
            streamer_text_colour = (85,)
            ai_text_colour = (255,)

        cv2.putText(wasd_im, 'Random', (25, 50), font, 1, streamer_text_colour, 2)
        cv2.putText(wasd_im, 'AI', (25, 100), font, 1, ai_text_colour, 2)
        cv2.putText(wasd_im, 'Reward: {:.2f}'.format(reward), (25, 200), font, 1, (255,), 2)
        cv2.putText(wasd_im, 'AI Action: {}'.format(letters[np.argmax(q_values)]), (25, 150), font, 1, (255,), 2)

        cv2.imshow('WASD', wasd_im)
        cv2.waitKey(1)

        cv2.putText(wasd_im, 'Reward: {:.2f}'.format(reward), (25, 200), font, 1, (0,), 2)
        cv2.putText(wasd_im, 'AI Action: {}'.format(letters[np.argmax(q_values)]), (25, 150), font, 1, (0,), 2)


class StateHistory:
    def __init__(self, history_len, mode):
        self.history_len = history_len  # max number of distinct frame_stacks in buffer

        array_path = data_path + '_frame_array'
        if mode == 'train':
            self.index = 0
            self.frames_stored = 0
            self.array_len = self.history_len + stack_len
            self.shape = (self.array_len, 90, 160)
            self.array = np.memmap(array_path, shape=(self.array_len, 90, 160), dtype=np.uint8, mode='w+')
        elif mode == 'resume':
            with open(data_path + '_history_state', 'rb') as state_file:
                history_state = pickle.load(state_file)
                self.index = history_state['index']
                self.frames_stored = history_state['frames_stored']
                self.shape = history_state['shape']
                self.array_len = self.shape[0]

            self.array = np.memmap(array_path, shape=self.shape, dtype=np.uint8, mode='r+')
        else:
            self.index = 0
            self.frames_stored = 0
            self.array_len = 10
            self.array = np.zeros((self.array_len, 90, 160), dtype=np.uint8)

    def append_frame(self, frame):
        self.array[self.index] = frame
        self.index = (self.index + 1) % self.array_len
        self.frames_stored = min(self.frames_stored + 1, self.array_len)

    def __getitem__(self, stack_index):
        # The array of frames is like a normal list up to a certain length, after which it acts like a circular buffer.
        # I.e. initially array[0] is the least recently added frame, but after array_len frames have been added
        # array[index - 1] is the least recently added frame. This is how the other components of the replay buffer
        # are used.
        start = (self.index - self.frames_stored + self.array_len + stack_index) % self.array_len
        end = (start + stack_len) % self.array_len

        if end < start:
            return np.concatenate((self.array[start:], self.array[:end]))  # x[:0] == []
        else:
            return self.array[start:end]

    def most_recent(self):
        start = (self.index - stack_len + self.array_len) % self.array_len
        end = (start + stack_len) % self.array_len

        if end < start:
            return np.concatenate((self.array[start:], self.array[:end]))
        else:
            return self.array[start:end]

    def save(self):
        self.array.flush()
        with open(data_path + '_history_state', 'wb') as state_file:
            history_state = {'index': self.index, 'shape': self.shape, 'frames_stored': self.frames_stored}
            pickle.dump(history_state, state_file)


class Env:
    def __init__(self, mode, history_len=10):
        self.state_history = StateHistory(history_len, mode)

        self.heading_number = 0

        self.sock = socket()
        while True:
            try:
                self.sock.connect(("127.0.0.1", 7001))
                break
            except ConnectionRefusedError:
                print("ConnectionRefusedError")
                sleep(1)

    def reset(self):
        self.sock.sendall(reinforcement_resumed_bytes)  # set control normals to 0
        self.sock.sendall(teleport_bytes)

        heading_bytes = heading_to_bytes(headings[self.heading_number])
        self.sock.sendall(heading_bytes)
        self.heading_number = (self.heading_number + 1) % num_headings

        # Don't append frames here otherwise the frames get misaligned from the rewards
        # self.state_history.append_frame(get_frame())

        return self.state_history.most_recent()

    def send_action(self, action):
        # w nk s
        match action:
            case 0:
                self.sock.sendall(w_bytes)
            case 1:
                self.sock.sendall(nk_bytes)
            case 2:
                self.sock.sendall(s_bytes)
            case _:
                print('Invalid action: {}'.format(action))

    def step(self, action, keys):
        reward = 0
        terminated = False
        action, _ = apply_correction(keys, action)

        self.send_action(action)
        recv_bytes = self.sock.recv(29)
        forward_speed = np.frombuffer(recv_bytes, dtype=np.float32, count=1, offset=1)
        forward_speed = forward_speed[0]

        # all_wheels_on_road = recv_bytes[5] == 4 and recv_bytes[6] == 4 and recv_bytes[7] == 4 and recv_bytes[8] == 4

        # I think this is the submerged check
        # if recv_bytes[9] == 1:
        #     terminated = True
        #     reward = -1
        # elif all_wheels_on_road:
        #     reward = max(0, forward_speed)

        # reward = floor(max(0, forward_speed))
        reward = abs(forward_speed)

        self.state_history.append_frame(get_frame())

        return self.state_history.most_recent(), reward, terminated

    def pause(self):
        self.sock.sendall(reinforcement_paused_bytes)

    def resume(self):
        self.sock.sendall(reinforcement_resumed_bytes)

    def close(self):
        self.sock.sendall(reinforcement_paused_bytes)
        self.sock.close()
