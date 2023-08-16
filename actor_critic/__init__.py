import os

import numpy as np

from common import MODEL_DIR

format_name = 'actor_critic'
model_file_name = os.path.join(MODEL_DIR, format_name)

# w_bytes = bytes([0]) + np.array([1, 0, 0, 0], dtype=np.float32).tobytes()
# wa_bytes = bytes([0]) + np.array([1, 1, 0, 0], dtype=np.float32).tobytes()
# wd_bytes = bytes([0]) + np.array([1, 0, 0, 1], dtype=np.float32).tobytes()
# s_bytes = bytes([0]) + np.array([0, 0, 1, 0], dtype=np.float32).tobytes()
resume_bytes = bytes([3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
pause_bytes = bytes([4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
position_bytes = bytes([15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
reinforcement_pause_bytes = bytes([5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])


def set_controls(sock, action_num):
    if action_num == 0:
        sock.sendall(w_bytes)
    elif action_num == 1:
        sock.sendall(s_bytes)
        # sock.sendall(wa_bytes)
    # elif action_num == 2:
    #     sock.sendall(wd_bytes)
    # elif action_num == 3:
    #     sock.sendall(s_bytes)
