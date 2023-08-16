import numpy as np

from common import get_keys, reinforcement_resumed_bytes, reinforcement_paused_bytes
from actor_critic.train import InputCollector

input_collector = InputCollector()

input_collector.sock.sendall(reinforcement_resumed_bytes)

while True:
    state = input_collector.return_inputs()

    print(state.numpy()[0])

    keys = get_keys()
    wasd = ['I' in keys, 'J' in keys, 'K' in keys, 'L' in keys]
    wasd_bytes = bytes([0]) + np.array(wasd, dtype=np.float32).tobytes()
    input_collector.sock.sendall(wasd_bytes)

    input_collector.update_inputs()

    if '5' in keys:
        input_collector.sock.sendall(reinforcement_paused_bytes)
        input_collector.sock.close()
        break
