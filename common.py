import os
from pathlib import Path
from time import time, sleep

import cv2
import numpy as np

# How to reload a module in an interactive session:
# import common
# import importlib
# importlib.reload(common)

# OpenCV windows get stuck to the sides of the screen and can open off-screen after a monitor configuration change.
# cv2.moveWindow('Figure 1', 1300, 400)

# A variable that is only referenced within a function is assumed to be global, whereas a variable which is
# assigned/modified is assumed to be local unless it is declared to be global.

# The nonlocal keyword allows a function to rebind a name which was previously bound in the function's scope. 
# I don't understand how this is different from the global keyword.

# https://youtrack.jetbrains.com/issue/PY-54649/OpenCV-code-completion-is-still-not-working#focus=Comments-27-6172519.0-0
# Used workaround #2 on this link to get rid of cv2 "cannot find reference" warnings (add cv2 package dir to interpreter
# paths for the interpreter in use).

# Simplest way of getting conv output is to make model with q_val output then another model with
# q_val and model.get_layer(index=2) outputs

# For Atari ROMS:
# pip install gymnasium[atari,accept-rom-license]
# This installs autorom.

np.set_printoptions(precision=3, floatmode='fixed', suppress=True, sign=' ')


INITIAL_DATA_FILE_NAME = 'initial_data_uint8.npy'
CORRECTION_DATA_FILE_NAME = 'correction_data_uint8.npy'
PAUSE_KEY = 'Z'
SAVE_AND_QUIT_KEY = 'X'
SAVE_AND_CONTINUE_KEY = 'B'
QUIT_WITHOUT_SAVING_KEY = '5'
KEYS = ['W', 'A', 'S', 'D']
CORRECTION_KEYS = ['I', 'J', 'K', 'L']
FORWARD = CORRECTION_KEYS[0]
LEFT = CORRECTION_KEYS[1]
BRAKE = CORRECTION_KEYS[2]
RIGHT = CORRECTION_KEYS[3]

DISPLAY_WIDTH = 960
DISPLAY_HEIGHT = 540


def get_save_path(env_name, suffix):
    return os.path.join(Path.home(), 'Data', env_name, suffix)


# Generic version of this function for use in experimental scripts
def imshow(data_param, width=750, height=None, frame_rate=0, title='imshow'):
    if height is None:
        height = data_param.image[0].shape[0] * round(width / data_param.image[0].shape[1])

    names = data_param.dtype.names

    for datum in data_param:
        st = time()
        for name, field in zip(names, datum):
            if field.ndim > 1:
                field = cv2.resize(field, (width, height), interpolation=cv2.INTER_NEAREST)
                cv2.imshow(title, field)
            else:
                print(name, field)

        if frame_rate == 0:
            if cv2.waitKey(0) == ord('q'):
                cv2.destroyAllWindows()
                return
        else:
            if cv2.waitKey(25) == ord('q'):
                cv2.destroyAllWindows()
                return
            sleep(max(0, round(1/frame_rate - (time() - st))))

    cv2.destroyAllWindows()


def uint8_to_float32(array):
    return array.astype(np.float32) / 255


def resize(im, width=None, height=None):
    if width is not None and height is not None:
        resize_width = width
        resize_height = height
    else:
        im_width = im.shape[1]
        im_height = im.shape[0]

        if width is not None:
            resize_width = width
            resize_height = int(im_height * (width / im_width))
        elif height is not None:
            resize_height = height
            resize_width = int(im_width * (height / im_height))
        else:
            raise ValueError('One of width or height has to be not None')

    return cv2.resize(im, (resize_width, resize_height), interpolation=cv2.INTER_NEAREST)


def put_text(image, text, position=(50, 50), font=cv2.FONT_HERSHEY_SIMPLEX, colour=(255, 255, 255), size=1, thickness=1):
    cv2.putText(image, text, position, font, size, colour, thickness)


class CircularBuffer:
    def __init__(self, array):
        self.array = array
        self.index = 0
        self.length = len(array)

    def append(self, item):
        if isinstance(item, zip):  # len not defined for zip
            item = list(item)

        if isinstance(item, list):
            item_len = len(item)

            if self.index + item_len > self.length:
                first_half_elem_num = self.length - self.index
                second_half_elem_num = item_len - first_half_elem_num
                self.array[self.index:] = item[:first_half_elem_num]
                self.array[:second_half_elem_num] = item[first_half_elem_num:]
            else:
                self.array[self.index:(self.index + item_len)] = item

            self.index = (self.index + item_len) % self.length

        else:
            self.array[self.index] = item
            self.index = (self.index + 1) % self.length

    def flush(self):
        self.array.flush()
