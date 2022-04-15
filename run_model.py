import cv2
from time import time, sleep
# from tensorflow.keras.models import load_model, Model

from common import (PAUSE_KEY, release_keys, QUIT_WITHOUT_SAVING_KEY, get_keys, CORRECTING_KEYS, INPUT_HEIGHT,
                    INPUT_WIDTH, ReleaseKey, S, PressKey, W)

from multihot_3 import get_frame, correction_to_keypresses, MODEL_NAME, ModelRunner

# start_time = 0

# model = load_model(MODEL_NAME)
# model = Model(inputs=model.inputs, outputs=[layer.output for layer in model.layers])
# prediction = []

model_runner = ModelRunner()

correcting = False
paused = True
get_keys()  # Flush key presses
print('Press {} to unpause'.format(PAUSE_KEY))

while True:
    keys = get_keys()

    if PAUSE_KEY in keys:
        if paused:
            paused = False
            print('Unpaused')
            sleep(1)
        else:
            paused = True
            release_keys()
            print('Paused')
            sleep(1)
    elif set(CORRECTING_KEYS) & set(keys):
        if not correcting and not paused:
            correcting = True
    elif QUIT_WITHOUT_SAVING_KEY in keys:
        print('Quitting')
        cv2.destroyAllWindows()
        release_keys()
        break
    else:
        if correcting:
            correcting = False

    if not paused:
        if correcting:
            correction_to_keypresses(keys)

        model_runner.run_model(keys, correcting)
