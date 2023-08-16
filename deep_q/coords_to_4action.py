from socket import socket
from time import sleep
import os

import numpy as np

from common import reinforcement_resumed_bytes, w_bytes, s_bytes, wa_bytes, wd_bytes, state_bytes, \
    reinforcement_paused_bytes

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from tensorflow import keras
from keras import layers

model_name = __name__
target_model_name = __name__ + '_target'


starting_position = np.array([1137.6, -409.4, 66.5], dtype=np.float32)
start_pos_net_coords = starting_position / 1000
teleport_bytes = bytes([19, 0, 0, 0]) + starting_position.tobytes() + bytes([0])
heading_array = np.zeros((3,), dtype=np.float32)
headings = [0, 45, 90, 135, 180, 225, 270, 315]
num_headings = len(headings)


def apply_correction(keys, action):
    if 'J' in keys:
        action = 2
    elif 'L' in keys:
        action = 3
    elif 'I' in keys:
        action = 0
    elif 'K' in keys:
        action = 1

    return action


class Env:
    def __init__(self):
        self.num_actions = 4

        self.heading_num = 0
        self.prev_distance = 0

        self.sock = socket()
        while True:
            try:
                self.sock.connect(("127.0.0.1", 7001))
                break
            except ConnectionRefusedError:
                print("ConnectionRefusedError")
                sleep(1)

    # num_actions could be a class variable because it's never modified during training, so create_q_model could then
    # be a @classmethod. But then there is only ever one instance of this class so arguably everything should be class
    # methods and class variables. That would look weird though, so I've decided (once again) to just have everything
    # as instance methods and instance variables. A @staticmethod appears to be a normal function which is defined
    # in a class which can be called like a method but does not reference a class or instance object.
    def create_q_model(self):
        inputs = layers.Input(shape=(7,))
        layer1 = layers.Dense(256, activation='relu')(inputs)
        action = layers.Dense(self.num_actions, activation="linear")(layer1)

        return keras.Model(inputs=inputs, outputs=action)

    def reset(self):
        self.sock.sendall(reinforcement_resumed_bytes)  # set control normals to 0
        self.sock.sendall(teleport_bytes)
        heading_array[2] = headings[self.heading_num]
        heading_bytes = bytes([21, 0, 0, 0]) + heading_array.tobytes() + bytes([0])
        self.sock.sendall(heading_bytes)
        self.heading_num = (self.heading_num + 1) % num_headings
        self.prev_distance = 0
        sleep(1)  # allow teleport to finish

        heading_state = heading_array[2:3] / 180 - 1
        position_state = starting_position / 1000
        velocity_state = np.zeros_like(starting_position)

        state = np.concatenate((heading_state, position_state, velocity_state))
        return state

    def step(self, action, keys):
        apply_correction(keys, action)

        if action == 0:
            self.sock.sendall(w_bytes)
        elif action == 1:
            self.sock.sendall(s_bytes)
        elif action == 2:
            self.sock.sendall(wa_bytes)
        elif action == 3:
            self.sock.sendall(wd_bytes)

        self.sock.sendall(state_bytes)
        recv_bytes = self.sock.recv(29)
        heading = np.frombuffer(recv_bytes, dtype=np.float32, count=1, offset=1) / 180 - 1  # -1.0 to +1.0
        position = np.frombuffer(recv_bytes, dtype=np.float32, count=3, offset=5) / 1000
        velocity = np.frombuffer(recv_bytes, dtype=np.float32, count=3, offset=17) / 100

        distance = np.linalg.norm(position - start_pos_net_coords)
        reward = (distance - self.prev_distance) * 1000
        self.prev_distance = distance

        state = np.concatenate((heading, position, velocity))

        return state, reward, False

    def quit(self):
        self.sock.sendall(reinforcement_paused_bytes)
        self.sock.close()
