import ctypes

import cv2
import numpy as np
import win32gui
import win32ui
import win32con
import win32api


window_width = 1280
window_height = 720
RESIZE_WIDTH = 160
RESIZE_HEIGHT = 90
INPUT_WIDTH = RESIZE_WIDTH
INPUT_HEIGHT = RESIZE_HEIGHT
# WINDOW_NAME = 'Grand Theft Auto V'
# WINDOW_NAME = 'FiveM® by Cfx.re - FXServer, but unconfigured'
WINDOW_NAME = 'FiveM® by Cfx.re - Ljbw City'
SCANCODES = [0x17, 0x24, 0x25, 0x26]
I = SCANCODES[0]
J = SCANCODES[1]
K = SCANCODES[2]
L = SCANCODES[3]
W = 0x11
A = 0x1E
S = 0x1F
D = 0x20


# cv2.resize wants (width, height) but we store (..., height, width, ...) in a numpy array shape
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

    signed_ints_array = bmp.GetBitmapBits(True)
    img = np.fromstring(signed_ints_array, dtype='uint8')
    img.shape = (height, width, 4)

    srcdc.DeleteDC()
    memdc.DeleteDC()
    win32gui.ReleaseDC(hwin, hwindc)
    win32gui.DeleteObject(bmp.GetHandle())

    return img


def get_frame(width=160, height=90, grey=True):
    frame = get_gta_window()

    if grey:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2GRAY)

    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_NEAREST)
    return frame


# C struct redefinitions
PUL = ctypes.POINTER(ctypes.c_ulong)


class KeyBoardInput(ctypes.Structure):
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


class InputI(ctypes.Union):
    _fields_ = [("ki", KeyBoardInput),
                ("mi", MouseInput),
                ("hi", HardwareInput)]


class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", InputI)]


SendInput = ctypes.windll.user32.SendInput


def press_key(hex_key_code):
    extra = ctypes.c_ulong(0)
    ii_ = InputI()
    ii_.ki = KeyBoardInput(0, hex_key_code, 0x0008, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(1), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))


def release_key(hex_key_code):
    extra = ctypes.c_ulong(0)
    ii_ = InputI()
    ii_.ki = KeyBoardInput(0, hex_key_code, 0x0008 | 0x0002, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(1), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))


def release_keys():
    release_key(W)
    release_key(A)
    release_key(S)
    release_key(D)


def get_keys():
    keys = []
    for key in "ABCDEFGHIJKLMNOPQRSTUVWXYZ 1234567890":
        if win32api.GetAsyncKeyState(ord(key)):
            keys.append(key)
    return keys
