import numpy as np
import cv2
from time import time, sleep
from tensorflow.keras.models import load_model, Model

from common import (PAUSE_KEY, release_keys, QUIT_WITHOUT_SAVING_KEY, get_keys, CORRECTING_KEYS, INPUT_HEIGHT,
                    INPUT_WIDTH)

from multihot_3 import get_frame, correction_to_keypresses, prediction_to_key_presses, display_features, predictions

# np.set_printoptions(precision=3)
start_time = 0

model = load_model('multihot_3')
model = Model(inputs=model.inputs, outputs=[layer.output for layer in model.layers])
prediction = []

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
        start_time = time()

        frame = get_frame()
        frame = frame.reshape(1, INPUT_HEIGHT, INPUT_WIDTH, 3)  # frame = np.expand_dims(frame, 0) ?

        if correcting:
            correction_to_keypresses(keys)
        else:
            prediction = model(frame)  # model returns a list of tensors which are the outputs of each layer
            predictions = prediction_to_key_presses(prediction[-1][0], predictions)

        display_features(prediction, correcting=correcting)

        # duration = time() - start_time
        # sleep(max(0., round(1/18 - duration)))
