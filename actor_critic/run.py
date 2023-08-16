from socket import socket
from time import sleep, time

import tensorflow.keras as ks
import numpy as np

from common import reinforcement_model_resumed_bytes, model_paused_bytes
from actor_critic import model_file_name, set_controls
from actor_critic.train import InputCollector, teleport_bytes, heading_bytes


class ModelRunner:
    def __init__(self, q_param):
        self.input_collector = InputCollector()
        self.model = ks.models.load_model(model_file_name)

        self.input_collector.sock.sendall(reinforcement_model_resumed_bytes)
        self.input_collector.sock.sendall(teleport_bytes)
        self.input_collector.sock.sendall(heading_bytes)

        self.paused = False

    def run_model(self, k, c):
        state = self.input_collector.return_inputs()

        action_probs, _ = self.model(state)

        print('state {}'.format(state.numpy()[0]))
        print(action_probs.numpy()[0])

        # action = np.random.choice(num_actions, p=np.squeeze(action_probs))
        action = np.argmax(np.squeeze(action_probs))

        set_controls(self.input_collector.sock, action)
        sleep(0.25)  # give action time to take affect

        self.input_collector.update_inputs()

    def pause_or_unpause(self, paused):
        if paused:
            self.input_collector.sock.sendall(model_paused_bytes)
        else:
            self.input_collector.sock.sendall(reinforcement_model_resumed_bytes)

        self.paused = paused

    def quit_model(self):
        self.input_collector.sock.sendall(model_paused_bytes)
        self.input_collector.sock.close()
