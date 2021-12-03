import numpy as np
import cv2
import time
import os
import win32api

from common import (INPUT_WIDTH, INPUT_HEIGHT, get_gta_window,
                    PAUSE_KEY, QUIT_AND_SAVE_KEY, model, MODEL_NAME, DATA_FILE_NAME,
                    W, A, S, D, PressKey, ReleaseKey, release_keys, CORRECTING_KEYS,
                    QUIT_WITHOUT_SAVING_KEY, OUTPUT_LENGTH, DISPLAY_WIDTH,
                    DISPLAY_HEIGHT, output_row, RESIZE_WIDTH, RESIZE_HEIGHT, w, a, s, d, wa, wd, sa, sd, nk)

def key_check():
    keys = []
    for key in "ABCDEFGHIJKLMNOPQRSTUVWXYZ 123456789":
        if win32api.GetAsyncKeyState(ord(key)):
            keys.append(key)
    return keys

model_exists = os.path.isfile('{}.index'.format(MODEL_NAME))
if model_exists:
    print('Loading previously trained model')
    model.load(MODEL_NAME)

previous_data_exists = os.path.isfile(DATA_FILE_NAME)
if previous_data_exists:
    print('Loading previously collected data')
    training_data = np.load(DATA_FILE_NAME)

np.set_printoptions(precision=3)
start_time = 0
output = np.zeros(OUTPUT_LENGTH,dtype='uint8')
new_data = np.empty((1000,INPUT_HEIGHT + 1,INPUT_WIDTH),dtype='uint8')
new_data_counter = 0
forwards = new_data.copy()
forwards_counter = 0
forwards_populated = False
default_rng = np.random.default_rng()
planned_outputs = np.zeros((4,OUTPUT_LENGTH),dtype='uint8')
planned_outputs_counter = 0

correcting = False
paused = True
print('Press {} to unpause'.format(PAUSE_KEY))
key_check() # Flush key presses

while True:
    keys = key_check()

    if PAUSE_KEY in keys:
        if paused:
            paused = False
            print('Unpaused')
            time.sleep(1)
        else:
            release_keys()
            paused = True
            print('Paused')
            print('new_data_counter',new_data_counter)
            if previous_data_exists:
                print('Training data at {} frames'.format(len(training_data)))
            time.sleep(1)
    elif set(CORRECTING_KEYS) & set(keys):
        if not correcting and not paused:
            correcting = True
    elif QUIT_AND_SAVE_KEY in keys:
        if previous_data_exists:
            training_data = np.concatenate((training_data, new_data[:new_data_counter]))
        else: # This allows us to save new data before it reaches 1000 frames
            training_data = new_data[:new_data_counter]
        print('Saving at {} frames of training data'.format(len(training_data)))
        np.save(DATA_FILE_NAME, training_data)
        cv2.destroyAllWindows()
        release_keys()
        break
    elif QUIT_WITHOUT_SAVING_KEY in keys:
        print('Not saving correction data')
        cv2.destroyAllWindows()
        release_keys()
        break
    else:
        if correcting and model_exists: # If model doesn't exist then stay in correcting state, to capture no-key's
            correcting = False

    if not paused:
        start_time = time.time()

        frame = get_gta_window()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # resize to something a bit more acceptable for a CNN
        frame = cv2.resize(frame, (RESIZE_WIDTH, RESIZE_HEIGHT))
        # frame.shape == (RESIZE_HEIGHT + OUTPUT_LENGTH, RESIZE_WIDTH)

        cv2.imshow('ALANN', cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation=cv2.INTER_NEAREST))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

        if model_exists:
            current_output = model.predict([frame.reshape(INPUT_WIDTH, INPUT_HEIGHT, 1)])[0]
            index = np.argmax(current_output)
            current_output[:] = 0
            current_output[index] = 1
            planned_outputs[planned_outputs_counter] = current_output
            # Set output to the prediction from four frames ago. This is so that there is symmetry
            # between how the model was trained and how it is driving.
            output = planned_outputs[(planned_outputs_counter + 3) % 4]
            planned_outputs_counter += 1
            planned_outputs_counter %= 4
        else:
            output = nk.copy()

        # If the prediction is saying the opposite of the correction then we
        # need to undo the prediction in addition to applying the correction
        if CORRECTING_KEYS[0] in keys:
            if output[1] or output[4] or output[6]: # if a or wa or sa
                output = wa.copy()
            elif output[3] or output[5] or output[7]: # if d or wd or sd
                output = wd.copy()
            else:
                output = w.copy()

        if CORRECTING_KEYS[1] in keys:
            if output[0] or output[4]: # if w or wa
                output = wa.copy()
            elif output[3] or output[5]: # if d or wd
                output = wa.copy()
            elif output[6] or output[7]: # if sa or sd
                output = sa.copy()
            else:
                output = a.copy()

        if CORRECTING_KEYS[2] in keys:
            if output[1] or output[4] or output[6]: # if a or wa or sa
                output = sa.copy()
            elif output[3] or output[5] or output[7]: # if d or wd or sd
                output = sd.copy()
            else:
                output = s.copy()

        if CORRECTING_KEYS[3] in keys:
            if output[0] or output[4]:  # if w or wa
                output = wd.copy()
            elif output[3] or output[5]:  # if d or wd
                output = wd.copy()
            elif output[2] or output[6] or output[7]:  # if s or sa or sd
                output = sd.copy()
            else:
                output = d.copy()

        if np.argmax(output) == 0:
            ReleaseKey(A)
            ReleaseKey(S)
            ReleaseKey(D)
            PressKey(W)
        elif np.argmax(output) == 1:
            ReleaseKey(W)
            ReleaseKey(S)
            ReleaseKey(D)
            PressKey(A)
        elif np.argmax(output) == 2:
            ReleaseKey(W)
            ReleaseKey(A)
            ReleaseKey(D)
            PressKey(S)
        elif np.argmax(output) == 3:
            ReleaseKey(W)
            ReleaseKey(A)
            ReleaseKey(S)
            PressKey(D)
        elif np.argmax(output) == 4:
            ReleaseKey(S)
            ReleaseKey(D)
            PressKey(W)
            PressKey(A)
        elif np.argmax(output) == 5:
            ReleaseKey(A)
            ReleaseKey(S)
            PressKey(W)
            PressKey(D)
        elif np.argmax(output) == 6:
            ReleaseKey(W)
            ReleaseKey(D)
            PressKey(A)
            PressKey(S)
        elif np.argmax(output) == 7:
            ReleaseKey(W)
            ReleaseKey(A)
            PressKey(S)
            PressKey(D)
        elif np.argmax(output) == 8:
            ReleaseKey(W)
            ReleaseKey(A)
            ReleaseKey(S)
            ReleaseKey(D)

        if correcting:
            output_row[0,:OUTPUT_LENGTH] = output
            frame = np.concatenate((frame, output_row))

            if np.array_equal(output,w):
                forwards[forwards_counter] = frame
                forwards_counter += 1
                if forwards_counter == 1000:
                    forwards_populated = True
                forwards_counter %= 1000
            else:
                new_data[new_data_counter] = frame
                new_data_counter += 1
                if forwards_populated and not (new_data_counter == 1000):
                    new_data[new_data_counter] = forwards[default_rng.integers(1000)]
                    new_data_counter += 1

            if new_data_counter == 1000:
                if previous_data_exists:
                    training_data = np.concatenate((training_data, new_data))
                    print('Training data at {} frames'.format(len(training_data)))
                    new_data_counter = 0
                else:
                    print('Training data at 1000 frames')
                    training_data = new_data.copy()
                    new_data_counter = 0
                    previous_data_exists = True
                    print(training_data.dtype)

        duration = time.time() - start_time
        time.sleep(max(0, 1/18 - duration))

