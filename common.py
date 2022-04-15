import numpy as np
from numpy import array_equal as ae
import platform

# How to reload a module in an interactive session:
# import importlib
# importlib.reload(common)

# np.set_printoptions(precision=3)

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
            # rect = win32gui.GetClientRect(hwin)
            left = CAPTURE_REGION[0] + 3  # left and top have to be offset by these values or we get
            top = CAPTURE_REGION[1] + 26  # pixels from outside the window. No idea why.
            width = CAPTURE_REGION[2]  # Don't need to add 1 to width & height.
            height = CAPTURE_REGION[3]

        hwindc = win32gui.GetWindowDC(hwin)
        srcdc = win32ui.CreateDCFromHandle(hwindc)
        memdc = srcdc.CreateCompatibleDC()
        bmp = win32ui.CreateBitmap()
        bmp.CreateCompatibleBitmap(srcdc, width, height)
        memdc.SelectObject(bmp)
        memdc.BitBlt((0, 0), (width, height), srcdc, (left, top), win32con.SRCCOPY)

        signedIntsArray = bmp.GetBitmapBits(True)
        img = np.fromstring(signedIntsArray, dtype='uint8')
        img.shape = (height, width, 4)

        srcdc.DeleteDC()
        memdc.DeleteDC()
        win32gui.ReleaseDC(hwin, hwindc)
        win32gui.DeleteObject(bmp.GetHandle())

        return img


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
        for key in "ABCDEFGHIJKLMNOPQRSTUVWXYZ 123456789":
            if win32api.GetAsyncKeyState(ord(key)):
                keys.append(key)
        return keys

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
# topleft_x, topleft_y, width, height
CAPTURE_REGION = (int(WINDOW_WIDTH / 4), int(WINDOW_HEIGHT / 4), int(WINDOW_WIDTH / 2), int(WINDOW_HEIGHT / 2))
RESIZE_WIDTH = 160
RESIZE_HEIGHT = 90
INPUT_WIDTH = RESIZE_WIDTH
INPUT_HEIGHT = RESIZE_HEIGHT
MODEL_NAME = 'Taxi'

WINDOW_NAME = 'Grand Theft Auto V'
INITIAL_DATA_FILE_NAME = 'initial_data_uint8.npy'
CORRECTION_DATA_FILE_NAME = 'correction_data_uint8.npy'
PAUSE_KEY = 'Z'
SAVE_AND_QUIT_KEY = 'X'
SAVE_AND_CONTINUE_KEY = 'B'
QUIT_WITHOUT_SAVING_KEY = '5'
KEYS = ['W', 'A', 'S', 'D']
CORRECTING_KEYS = ['I', 'J', 'K', 'L']
FORWARD = CORRECTING_KEYS[0]
LEFT = CORRECTING_KEYS[1]
BRAKE = CORRECTING_KEYS[2]
RIGHT = CORRECTING_KEYS[3]
DISPLAY_WIDTH = 960
DISPLAY_HEIGHT = 540


# def correction_keys_to_key_presses(keys):
#     release_keys()
#     if FORWARD in keys:
#         PressKey(W)
#
#     if LEFT in keys:
#         PressKey(A)
#
#     if BRAKE in keys:
#         PressKey(S)
#
#     if RIGHT in keys:
#         PressKey(D)


eye9 = np.eye(9, dtype='uint8')
w = eye9[0]
a = eye9[1]
s = eye9[2]
d = eye9[3]
wa = eye9[4]
wd = eye9[5]
sa = eye9[6]
sd = eye9[7]
nk = eye9[8]

mh_w = np.array([1, 0, 0, 0], dtype='float32')
mh_a = np.array([0, 1, 0, 0], dtype='float32')
mh_s = np.array([0, 0, 1, 0], dtype='float32')
mh_d = np.array([0, 0, 0, 1], dtype='float32')
mh_wa = np.array([1, 1, 0, 0], dtype='float32')
mh_wd = np.array([1, 0, 1, 0], dtype='float32')
mh_sa = np.array([0, 1, 1, 0], dtype='float32')
mh_sd = np.array([0, 0, 1, 1], dtype='float32')
mh_nk = np.array([0, 0, 0, 0], dtype='float32')


def oh_to_mh(array):
    if ae(array, w):
        return mh_w
    elif ae(array, a):
        return mh_a
    elif ae(array, s):
        return mh_s
    elif ae(array, d):
        return mh_d
    elif ae(array, wa):
        return mh_wa
    elif ae(array, wd):
        return mh_wd
    elif ae(array, sa):
        return mh_sa
    elif ae(array, sd):
        return mh_sd
    elif ae(array, nk):
        return mh_nk


oh4_w = np.array([1, 0, 0, 0], dtype='float32')
oh4_wa = np.array([0, 1, 0, 0], dtype='float32')
oh4_wd = np.array([0, 0, 1, 0], dtype='float32')
oh4_s = np.array([0, 0, 0, 1], dtype='float32')
