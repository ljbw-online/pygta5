""" AlexNet.
References:
    - Alex Krizhevsky, Ilya Sutskever & Geoffrey E. Hinton. ImageNet
    Classification with Deep Convolutional Neural Networks. NIPS, 2012.
Links:
    - [AlexNet Paper](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
"""

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import local_response_normalization

def alexnet(input_width, input_height, lr, output_length):
    network = input_data(shape=[None, input_width, input_height, 1], name='input')
    network = conv_2d(network, 96, 11, strides=4, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 256, 5, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 256, 3, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, output_length, activation='softmax')
    network = regression(
        network,
        optimizer='momentum',
        loss='categorical_crossentropy',
        learning_rate=lr,
        name='targets'
    )
    model = tflearn.DNN(
        network,
        # checkpoint_path='model_alexnet',
        # max_checkpoints=1,
        tensorboard_verbose=2,
        tensorboard_dir='log'
    )
    return model

import win32gui, win32ui, win32con
import numpy as np
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
import time

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

CAPTURE_REGION = (160,170,480,270) # topleft_x, topleft_y, width, height
RESIZE_WIDTH = 112
RESIZE_HEIGHT = 63
OUTPUT_LENGTH = 9
INPUT_WIDTH = RESIZE_WIDTH
INPUT_HEIGHT = RESIZE_HEIGHT + OUTPUT_LENGTH
LR = 1e-3
EPOCHS = 2
MODEL_NAME = 'Airtug'
model = alexnet(INPUT_WIDTH, INPUT_HEIGHT, LR, OUTPUT_LENGTH)

TURNING_THRESHOLD = 0.1
BRAKING_THRESHOLD = 0.1
FORWARD_THRESHOLD = 0.1
WINDOW_NAME = 'Grand Theft Auto V'
DATA_FILE_NAME = 'training_data.npy'
PAUSE_KEY = 'Z'
CORRECTING_KEYS = ['I','J','K','L']
QUIT_AND_SAVE_KEY = 'X'
QUIT_WITHOUT_SAVING_KEY = '5'
UPSCALE_FACTOR = 8 # imshow window should be 896 by 504
DISPLAY_WIDTH = INPUT_WIDTH * UPSCALE_FACTOR
DISPLAY_HEIGHT = INPUT_HEIGHT * UPSCALE_FACTOR
recent_keys = np.zeros((OUTPUT_LENGTH,INPUT_WIDTH), dtype='uint8')

eye9 = np.eye(9,dtype='uint8')
w = eye9[0]
a = eye9[1]
s = eye9[2]
d = eye9[3]
wa = eye9[4]
wd = eye9[5]
sa = eye9[6]
sd = eye9[7]
nk = eye9[8]