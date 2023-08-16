import os
from pathlib import Path
from time import time, sleep
from socket import socket
from threading import Thread

import numpy as np
import tensorflow
import tensorflow.keras as ks

from . import __name__ as format_name
from .collect_initial import InputCollector

np.set_printoptions(precision=3, floatmode='fixed', suppress=True)  # suppress stops scientific notation
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
# tensorflow.compat.v1.enable_eager_execution()  # Necessary for .numpy() in TensorFlow 1

MODEL_PATH = os.path.join(Path.home(), 'My Drive\\Models',  format_name)

layer_outputs = [0, 0, 0, 0]  # PyCharm appeasement. Still necessary to be at module level?


class ModelRunner(InputCollector):
    def __init__(self, q_param):
        super().__init__()
        # from tensorflow.keras.models import load_model, Model
        model = ks.models.load_model(MODEL_PATH)
        model = ks.models.Model(inputs=model.inputs, outputs=[model.get_layer('1st_conv').output,
                                                              model.get_layer('2nd_conv').output,
                                                              model.get_layer('penult_dense').output,
                                                              model.get_layer('model_output').output])
        self.model = model
        self.stuck = False
        self.stuck_time = 0
        self.dormant = False
        self.most_recent_not_coasting = time()

        self.q = q_param
        self.q_counter = 0

        self.sock = socket()

        self.bot_recv_thread = Thread(target=self.bot_recv)
        self.bot_recv_thread.start()
        self.paused = True

        while True:
            try:
                self.sock.connect(("127.0.0.1", 7001))
                break
            except ConnectionRefusedError:
                print("Connection Refused")
                sleep(1)

        # self.sock.sendall((0).to_bytes(1, byteorder='big', signed=False))

    def run_model(self, keys, correcting):
        global layer_outputs
        self.collect_input(keys)

        if not self.correction_data:
            for i in range(1, 4):
                if (self.current_key_durations[i]) > 3.0:
                    print('Resetting key score {}'.format(i))
                    self.current_key_durations[i] = 0
                    self.key_duration_mins[i] = 0
                    self.key_press_times[i] = self.t
                    self.key_duration_maxes[i] = 0

        if (not correcting) or self.correction_data:
            layer_outputs = self.model([np.expand_dims(self.current_frame.astype('float32') / 255, 0),
                                        np.expand_dims(np.array(self.current_key_scores, dtype='float32'), 0),
                                        np.expand_dims(np.array(self.previous_key_state, dtype='float32'), 0)])

            # prediction_and_correction_to_key_presses(layer_outputs[-1][0], keys, self.dormant)

            try:
                if time() - self.most_recent_not_coasting > 2.0:
                    self.sock.sendall(np.array([1, 0, 0, 0], dtype=np.float32).tobytes())
                else:
                    self.sock.sendall(layer_outputs[-1][0].numpy().tobytes())
            except ConnectionResetError:
                print("ConnectionResetError")
                while True:
                    self.sock = socket()
                    try:
                        self.sock.connect(("127.0.0.1", 7001))
                        break
                    except ConnectionRefusedError:
                        print("ConnectionRefusedError")
                        sleep(1)

                # print(layer_outputs[-1][0].numpy())

        # if not correcting:
        # elif self.correction_data:
        #     prediction_and_correction_to_key_presses(layer_outputs[-1][0], keys, self.dormant)

        # self.q_counter = (self.q_counter + 1) % 2
        # if self.q_counter == 0:
        #     self.q.put((self.current_key_state, self.current_key_durations, layer_outputs, correcting))

        if not self.correction_data:
            prev_output = layer_outputs[-1][0]
            if (prev_output[0] > 0.5) or (prev_output[2] > 0.5):
                self.dormant = False
                self.most_recent_not_coasting = time()
            else:
                self.dormant = True

            dormant_time = time() - self.most_recent_not_coasting
            if self.dormant:
                if (dormant_time > 3.0) and (dormant_time < 3.5) and ('I' not in keys):
                    print('dormant')
                    self.most_recent_not_coasting = time()
                    # PressKey(I)
                elif (dormant_time >= 1.0) and ('I' in keys):
                    pass
                    # ReleaseKey(I)

    def quit_model(self):
        self.q.put((None, None, None, None))
        # self.sock.sendall((1).to_bytes(1, byteorder='big', signed=False))  # Send zero to ljbw_bot
        try:
            self.sock.sendall(bytes([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
            self.sock.close()
            self.sock = None
            self.bot_recv_thread.join(0)
        except ConnectionResetError:
            pass

    def pause_or_unpause(self, paused):
        try:
            self.sock.sendall(bytes([paused, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
        except ConnectionResetError:
            pass

        self.paused = paused

    def bot_recv(self):
        while True:
            try:
                bot_msg = self.sock.recv(16)
                if bot_msg[0] < 2:
                    self.paused = bot_msg[0] == 1
            except ConnectionRefusedError:
                print("ConnectionRefusedError")
                sleep(1)
            except OSError:  # socket not connected yet
                sleep(1)

            if self.sock is None:
                break
