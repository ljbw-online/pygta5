import numpy as np
import cv2
import time
import random

from directkeys import PressKey,ReleaseKey, W, A, S, D
from getkeys import key_check
from common import INPUT_WIDTH, INPUT_HEIGHT, MODEL_NAME, model, get_gta_window, PAUSE_KEY, QUIT_AND_SAVE_KEY

model.load(MODEL_NAME)

threshold = 0.1

def main():
    last_time = time.time()

    paused = True
    print('Press {} to unpause'.format(PAUSE_KEY))
    while True:
        if not paused:
            # print('loop took {} seconds'.format(time.time()-last_time))
            last_time = time.time()

            screen = get_gta_window()
            screen = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
            screen = cv2.resize(screen, (INPUT_WIDTH, INPUT_HEIGHT))

            prediction = model.predict([screen.reshape(INPUT_WIDTH, INPUT_HEIGHT, 1)])[0]
            # print(prediction)

            PressKey(W)

            if prediction[1] > (1 - threshold) and prediction[3] < threshold:
                PressKey(A)
                print('A')
            else:
                ReleaseKey(A)

            if prediction[1] < threshold and prediction[3] > (1 - threshold):
                PressKey(D)
                print('D')
            else:
                ReleaseKey(D)

        keys = key_check()

        # p pauses game and can get annoying.
        if PAUSE_KEY in keys:
            if paused:
                paused = False
                time.sleep(1)
            else:
                paused = True
                ReleaseKey(W)
                ReleaseKey(A)
                ReleaseKey(S)
                ReleaseKey(D)
                time.sleep(1)
        elif QUIT_AND_SAVE_KEY in keys:
            ReleaseKey(W)
            ReleaseKey(A)
            ReleaseKey(S)
            ReleaseKey(D)
            break

main()
