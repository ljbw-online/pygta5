import os
from pathlib import Path
from time import time, sleep

import cv2
import numpy as np
import platform

# How to reload a module in an interactive session:
# import importlib
# importlib.reload(common)

# OpenCV windows get stuck to the sides of the screen and can open off-screen after a monitor configuration change.
# cv2.moveWindow('Figure 1', 1300, 400)

# A variable that is only referenced within a function is assumed to be global, whereas a variable which is
# assigned/modified is assumed to be local unless it is declared to be global.

# https://youtrack.jetbrains.com/issue/PY-54649/OpenCV-code-completion-is-still-not-working#focus=Comments-27-6172519.0-0
# Used workaround #2 on this link to get rid of cv2 "cannot find reference" warnings (add cv2 package dir to interpreter
# paths for the interpreter in use).

# Simplest way of getting conv output is to make model with q_val output then another model with
# q_val and model.get_layer(index=2) outputs

# FOR ATARI ROMS:
# pip install gymnasium[atari,accept-rom-license]
# This installs autorom which allows tf-agents.environments to load Atari games.

np.set_printoptions(precision=3, floatmode='fixed', suppress=True, sign=' ')

if platform.system() == 'Windows':
    import win32gui
    import win32ui
    import win32con


    def get_gta_window(region=None):
        if region:
            hwin = win32gui.GetDesktopWindow()
            left, top, x2, y2 = region
            width = x2 - left + 1
            height = y2 - top + 1
        else:
            hwin = win32gui.FindWindow(None, WINDOW_NAME)
            (left, top, right, bottom) = win32gui.GetWindowRect(hwin)
            # global window_width
            # global window_height
            # window_width = right - left
            # window_height = bottom - top
            # capture_region = (
            #     int(window_width / 4), int(window_height / 4), int(window_width / 2), int(window_height / 2))
            # left = capture_region[0]  # + 3  # left and top have to be offset by these values or we get
            # top = capture_region[1]  # + 26  # pixels from outside the window. No idea why.
            # width = capture_region[2]  # Don't need to add 1 to width & height.
            # height = capture_region[3]

            width = right - left
            height = bottom - top

        hwindc = win32gui.GetWindowDC(hwin)
        srcdc = win32ui.CreateDCFromHandle(hwindc)
        memdc = srcdc.CreateCompatibleDC()
        bmp = win32ui.CreateBitmap()
        bmp.CreateCompatibleBitmap(srcdc, width, height)
        memdc.SelectObject(bmp)
        memdc.BitBlt((0, 0), (width, height), srcdc, (0, 0), win32con.SRCCOPY)

        signedIntsArray = bmp.GetBitmapBits(True)
        img = np.fromstring(signedIntsArray, dtype='uint8')
        img.shape = (height, width, 4)

        srcdc.DeleteDC()
        memdc.DeleteDC()
        win32gui.ReleaseDC(hwin, hwindc)
        win32gui.DeleteObject(bmp.GetHandle())

        return img

    # cv2.resize wants (width, height) but we store (..., height, width, ...) in a numpy array shape
    def get_frame(width=160, height=90, grey=True):
        frame = get_gta_window()

        if grey:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2GRAY)

        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_NEAREST)
        return frame

    # source to this solution and code:
    # https://stackoverflow.com/questions/14489013/simulate-python-keypresses-for-controlling-a-game
    # https://www.gamespp.com/directx/directInputKeyboardScanCodes.html
    import ctypes

    SendInput = ctypes.windll.user32.SendInput

    W = 0x11
    A = 0x1E
    S = 0x1F
    D = 0x20

    # C struct redefinitions
    PUL = ctypes.POINTER(ctypes.c_ulong)


    class KeyBdInput(ctypes.Structure):
        _fields_ = [("wVk", ctypes.c_ushort),
                    ("wScan", ctypes.c_ushort),
                    ("dwFlags", ctypes.c_ulong),
                    ("time", ctypes.c_ulong),
                    ("dwExtraInfo", PUL)]


    class HardwareInput(ctypes.Structure):
        _fields_ = [("uMsg", ctypes.c_ulong),
                    ("wParamL", ctypes.c_short),
                    ("wParamH", ctypes.c_ushort)]


    class MouseInput(ctypes.Structure):
        _fields_ = [("dx", ctypes.c_long),
                    ("dy", ctypes.c_long),
                    ("mouseData", ctypes.c_ulong),
                    ("dwFlags", ctypes.c_ulong),
                    ("time", ctypes.c_ulong),
                    ("dwExtraInfo", PUL)]


    class Input_I(ctypes.Union):
        _fields_ = [("ki", KeyBdInput),
                    ("mi", MouseInput),
                    ("hi", HardwareInput)]


    class Input(ctypes.Structure):
        _fields_ = [("type", ctypes.c_ulong),
                    ("ii", Input_I)]


    # Actual Functions


    def PressKey(hexKeyCode):
        extra = ctypes.c_ulong(0)
        ii_ = Input_I()
        ii_.ki = KeyBdInput(0, hexKeyCode, 0x0008, 0, ctypes.pointer(extra))
        x = Input(ctypes.c_ulong(1), ii_)
        ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))


    def ReleaseKey(hexKeyCode):
        extra = ctypes.c_ulong(0)
        ii_ = Input_I()
        ii_.ki = KeyBdInput(0, hexKeyCode, 0x0008 | 0x0002, 0, ctypes.pointer(extra))
        x = Input(ctypes.c_ulong(1), ii_)
        ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))


    def release_keys():
        # for letter, scancode in zip(KEYS, [W, A, S, D]):
        #     if letter in keys:
        #         ReleaseKey(scancode)
        #
        ReleaseKey(W)
        ReleaseKey(A)
        ReleaseKey(S)
        ReleaseKey(D)


    import win32api


    def get_keys():
        keys = []
        for key in "ABCDEFGHIJKLMNOPQRSTUVWXYZ 1234567890":
            if win32api.GetAsyncKeyState(ord(key)):
                keys.append(key)
        return keys

window_width = 1280
window_height = 720
# topleft_x, topleft_y, width, height

RESIZE_WIDTH = 160
RESIZE_HEIGHT = 90
INPUT_WIDTH = RESIZE_WIDTH
INPUT_HEIGHT = RESIZE_HEIGHT
MODEL_NAME = 'Taxi'

# WINDOW_NAME = 'Grand Theft Auto V'
# WINDOW_NAME = 'FiveM® by Cfx.re - FXServer, but unconfigured'
WINDOW_NAME = 'FiveM® by Cfx.re - Ljbw City'
# WINDOW_NAME = 'FiveM® by Cfx.re - Diamonds Roleplay | Starter vehicle | Free gift | 40k starter money'
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
SCANCODES = [0x17, 0x24, 0x25, 0x26]
I = SCANCODES[0]
J = SCANCODES[1]
K = SCANCODES[2]
L = SCANCODES[3]

# I = 0x17
# J = 0x24
# K = 0x25
# L = 0x26

DISPLAY_WIDTH = 960
DISPLAY_HEIGHT = 540

MODEL_DIR = os.path.join(Path.home(), 'My Drive\\Models')

supervised_resumed_bytes = bytes([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
model_paused_bytes = bytes([2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
reinforcement_resumed_bytes = bytes([4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
reinforcement_paused_bytes = bytes([5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
reinforcement_model_resumed_bytes = bytes([3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
position_bytes = bytes([15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
state_bytes = bytes([23, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

w_bytes = bytes([22]) + np.array([1, 0, 0, 0], dtype=np.float32).tobytes()
wa_bytes = bytes([22]) + np.array([1, 1, 0, 0], dtype=np.float32).tobytes()
wd_bytes = bytes([22]) + np.array([1, 0, 0, 1], dtype=np.float32).tobytes()
s_bytes = bytes([22]) + np.array([0, 0, 1, 0], dtype=np.float32).tobytes()
sa_bytes = bytes([22]) + np.array([0, 1, 1, 0], dtype=np.float32).tobytes()
sd_bytes = bytes([22]) + np.array([0, 0, 1, 1], dtype=np.float32).tobytes()
a_bytes = bytes([22]) + np.array([0, 1, 0, 0], dtype=np.float32).tobytes()
d_bytes = bytes([22]) + np.array([0, 0, 0, 1], dtype=np.float32).tobytes()
nk_bytes = bytes([22]) + np.array([0, 0, 0, 0], dtype=np.float32).tobytes()


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


def put_text(image, text, position=(50, 50), font=cv2.FONT_HERSHEY_SIMPLEX, colour=(255, 255, 255), size=1, thickness=2):
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
