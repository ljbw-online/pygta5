import os
from pathlib import Path

from common import FORWARD, LEFT, BRAKE, RIGHT, PressKey, ReleaseKey, W, A, S, D

MODEL_PATH = os.path.join(Path.home(), 'My Drive\\Models',  __name__)
DATA_DIRECTORY = os.path.join(Path.home(), 'Documents\\Data')


def correction_to_keypresses(keys):
    if FORWARD in keys and 'W' not in keys:
        PressKey(W)
    elif FORWARD not in keys and 'W' in keys:
        ReleaseKey(W)

    if LEFT in keys and 'A' not in keys:
        PressKey(A)
    elif LEFT not in keys and 'A' in keys:
        ReleaseKey(A)

    if BRAKE in keys and 'S' not in keys:
        PressKey(S)
    elif BRAKE not in keys and 'S' in keys:
        ReleaseKey(S)

    if RIGHT in keys and 'D' not in keys:
        PressKey(D)
    elif RIGHT not in keys and 'D' in keys:
        ReleaseKey(D)
