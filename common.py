import numpy as np
from numpy import array_equal as ae
import platform

# How to reload a module in an interactive session:
# import importlib
# importlib.reload(common)

if platform.system() == 'Windows':
    import win32gui, win32ui, win32con
    def get_gta_window(region=None):
        if region:
            hwin = win32gui.GetDesktopWindow()
            left, top, x2, y2 = region
            width = x2 - left + 1
            height = y2 - top + 1
        else:
            hwin = win32gui.FindWindow(None, WINDOW_NAME)
            rect = win32gui.GetClientRect(hwin)
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
                    ("time",ctypes.c_ulong),
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
        ii_.ki = KeyBdInput( 0, hexKeyCode, 0x0008, 0, ctypes.pointer(extra) )
        x = Input( ctypes.c_ulong(1), ii_ )
        ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

    def ReleaseKey(hexKeyCode):
        extra = ctypes.c_ulong(0)
        ii_ = Input_I()
        ii_.ki = KeyBdInput( 0, hexKeyCode, 0x0008 | 0x0002, 0, ctypes.pointer(extra) )
        x = Input( ctypes.c_ulong(1), ii_ )
        ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

    def release_keys():
        ReleaseKey(W)
        ReleaseKey(A)
        ReleaseKey(S)
        ReleaseKey(D)

    import win32api
    def key_check():
        keys = []
        for key in "ABCDEFGHIJKLMNOPQRSTUVWXYZ 123456789":
            if win32api.GetAsyncKeyState(ord(key)):
                keys.append(key)
        return keys

CAPTURE_REGION = (160, 170, 480, 270)  # topleft_x, topleft_y, width, height
RESIZE_WIDTH = 112
RESIZE_HEIGHT = 63
OUTPUT_LENGTH = 4
INPUT_WIDTH = RESIZE_WIDTH
INPUT_HEIGHT = RESIZE_HEIGHT
LR = 1e-3
EPOCHS = 2
MODEL_NAME = 'Taxi'

TURNING_THRESHOLD = 0.1
BRAKING_THRESHOLD = 0.1
FORWARD_THRESHOLD = 0.1
WINDOW_NAME = 'Grand Theft Auto V'
INITIAL_DATA_FILE_NAME = 'initial_data_uint8.npy'
CORRECTION_FILE_NAME = 'correction_data.npy'
PAUSE_KEY = 'Z'
CORRECTING_KEYS = ['I', 'J', 'K', 'L']
SAVE_AND_QUIT_KEY = 'X'
SAVE_AND_CONTINUE_KEY = 'B'
QUIT_WITHOUT_SAVING_KEY = '5'
UPSCALE_FACTOR = 8  # imshow window should be 896 by 504
DISPLAY_WIDTH = INPUT_WIDTH * UPSCALE_FACTOR
DISPLAY_HEIGHT = INPUT_HEIGHT * UPSCALE_FACTOR
output_row = np.zeros((1, INPUT_WIDTH), dtype='float32')

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

mh_w = np.array([1,0,0,0],dtype='float32')
mh_a = np.array([0,1,0,0],dtype='float32')
mh_s = np.array([0,0,1,0],dtype='float32')
mh_d = np.array([0,0,0,1],dtype='float32')
mh_wa = np.array([1,1,0,0],dtype='float32')
mh_wd = np.array([1,0,1,0],dtype='float32')
mh_sa = np.array([0,1,1,0],dtype='float32')
mh_sd = np.array([0,0,1,1],dtype='float32')
mh_nk = np.array([0,0,0,0],dtype='float32')

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

def oh_stats(data):
    forwards = 0
    lefts = 0
    brakes = 0
    rights = 0
    forward_lefts = 0
    forward_rights = 0
    brake_lefts = 0
    brake_rights = 0
    nokeys = 0

    for keystate in data[:,-1,:OUTPUT_LENGTH]:
        if np.array_equal(keystate,w):
            forwards += 1
        elif np.array_equal(keystate,a):
            lefts += 1
        elif np.array_equal(keystate,s):
            brakes += 1
        elif np.array_equal(keystate,d):
            rights += 1
        elif np.array_equal(keystate,wa):
            forward_lefts += 1
        elif np.array_equal(keystate,wd):
            forward_rights += 1
        elif np.array_equal(keystate,sa):
            brake_lefts += 1
        elif np.array_equal(keystate,sd):
            brake_rights += 1
        elif np.array_equal(keystate,nk):
            nokeys += 1
        else:
            print('huh?',keystate)

    print('forwards',forwards)
    print('lefts',lefts)
    print('brakes',brakes)
    print('rights',rights)
    print('forward_lefts',forward_lefts)
    print('forward_rights',forward_rights)
    print('brake_lefts',brake_lefts)
    print('brake_rights',brake_rights)
    print('nokeys',nokeys)

    return [forwards,lefts,brakes,lefts,forward_lefts,forward_rights,brake_lefts,brake_rights,nokeys]


oh4_w = np.array([1, 0, 0, 0], dtype='float32')
oh4_wa = np.array([0, 1, 0, 0], dtype='float32')
oh4_wd = np.array([0, 0, 1, 0], dtype='float32')
oh4_s = np.array([0, 0, 0, 1], dtype='float32')
