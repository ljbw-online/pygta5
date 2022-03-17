import numpy as np
import cv2
from time import time, sleep
from tensorflow.keras.models import load_model

from common import (MODEL_NAME, get_gta_window, PAUSE_KEY, QUIT_WITHOUT_SAVING_KEY, PressKey,
                    W, A, S, D, get_keys, CORRECTING_KEYS, DISPLAY_WIDTH, DISPLAY_HEIGHT, release_keys)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

INPUT_WIDTH = 112
INPUT_HEIGHT = 63
OUTPUT_LENGTH = 4

model = load_model(MODEL_NAME)
output = np.zeros(OUTPUT_LENGTH, dtype='bool')

last_loop_time = 0
last_correcting_time = 0
paused = True
correcting = False
get_keys()  # flush keys

print('Press {} to unpause'.format(PAUSE_KEY))
while True:
    keys = get_keys()

    if PAUSE_KEY in keys:
        if paused:
            paused = False
            sleep(1)
        else:
            paused = True
            release_keys()
            sleep(1)
    elif set(CORRECTING_KEYS) & set(keys):
        if not correcting and not paused:
            correcting = True
    elif QUIT_WITHOUT_SAVING_KEY in keys:
        release_keys()
        cv2.destroyAllWindows()
        break
    else:
        if correcting:
            correcting = False

    if not paused:
        last_loop_time = time()

        frame = get_gta_window()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (INPUT_WIDTH, INPUT_HEIGHT))

        display_frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation=cv2.INTER_NEAREST)

        frame = frame.astype('float32') / 255.0

        if (time() - last_correcting_time) < 1:
            text_top_left = (round(DISPLAY_WIDTH*0.1), round(DISPLAY_HEIGHT*0.9))
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(display_frame, 'Under human control', text_top_left, font, 2, (255, 255, 255), 3)

        cv2.imshow('ALANN', display_frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

        output[:] = False
        if correcting:
            last_correcting_time = time()

            if CORRECTING_KEYS[1] in keys:
                output[1] = True
            elif CORRECTING_KEYS[2] in keys:
                output[2] = True
            elif CORRECTING_KEYS[0] in keys:
                output[0] = True
            elif CORRECTING_KEYS[3] in keys:
                output[3] = True

        else:
            prediction = model(frame.reshape(1, INPUT_HEIGHT, INPUT_WIDTH, 1))

            # print(prediction.numpy()[0])

            argmax = np.argmax(prediction)
            output[argmax] = True

        release_keys()
        if output[0]:
            PressKey(W)
        elif output[1]:
            PressKey(W)
            PressKey(A)
        elif output[2]:
            PressKey(S)
        elif output[3]:
            PressKey(W)
            PressKey(D)

        # duration = time() - last_loop_time
        # sleep(max(0., 1/18 - duration))
