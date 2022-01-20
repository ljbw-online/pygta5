import numpy as np
import cv2
from time import time, sleep
from tensorflow.keras.models import load_model

from common import (INPUT_WIDTH, INPUT_HEIGHT, MODEL_NAME, get_gta_window, PAUSE_KEY, QUIT_WITHOUT_SAVING_KEY, PressKey,
                    W, A, S, D, key_check, CORRECTING_KEYS, DISPLAY_WIDTH, DISPLAY_HEIGHT, release_keys)

model = load_model(MODEL_NAME)
last_loop_time = 0
last_correcting_time = 0
paused = True
correcting = False
output = np.zeros(4, dtype='bool')
key_check()  # flush keys

print('Press {} to unpause'.format(PAUSE_KEY))
while True:
    keys = key_check()

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
            output[:] = False

    if not paused:
        last_loop_time = time()

        frame = get_gta_window()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (INPUT_WIDTH, INPUT_HEIGHT))
        frame = frame.astype('float32') / 255.0

        prediction = model.predict(frame.reshape(1, INPUT_HEIGHT, INPUT_WIDTH, 1))[0]

        if correcting:
            last_correcting_time = time()

        display_frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation=cv2.INTER_NEAREST)
        if (time() - last_correcting_time) < 1:
            text_top_left = (round(DISPLAY_WIDTH*0.1), round(DISPLAY_HEIGHT*0.9))
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(display_frame, 'Under human control', text_top_left, font, 2, (255, 255, 255), 3)

        cv2.imshow('ALANN', display_frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

        output[:] = False

        argmax = np.argmax(prediction)
        output[argmax] = True

        if correcting:

            if CORRECTING_KEYS[0] in keys:
                output[:] = False
                output[0] = True

            if CORRECTING_KEYS[1] in keys:
                output[:] = False
                output[1] = True

            if CORRECTING_KEYS[3] in keys:
                output[:] = False
                output[2] = True

            if CORRECTING_KEYS[2] in keys:
                output[:] = False
                output[3] = True

        release_keys()
        if output[0]:
            PressKey(W)
        elif output[1]:
            PressKey(W)
            PressKey(A)
        elif output[2]:
            PressKey(W)
            PressKey(D)
        elif output[3]:
            PressKey(S)

        # duration = time() - last_loop_time
        # sleep(max(0., 1/18 - duration))
