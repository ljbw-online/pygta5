import numpy as np
import cv2
import time
import os
import random
import win32api

from common import (INPUT_WIDTH, INPUT_HEIGHT, get_gta_window, WINDOW_NAME,
                    PAUSE_KEY, QUIT_AND_SAVE_KEY, model, MODEL_NAME, TURNING_THRESHOLD, DATA_FILE_NAME,
                    W, A, S, D, PressKey, ReleaseKey, release_keys, CORRECTING_KEYS, BRAKING_THRESHOLD,
                    FORWARD_THRESHOLD, QUIT_WITHOUT_SAVING_KEY, OUTPUT_LENGTH, DISPLAY_WIDTH,
                    DISPLAY_HEIGHT, recent_keys, RESIZE_WIDTH, RESIZE_HEIGHT, w,a,s,d,wa,wd,sa,sd,nk)

def key_check():
    keys = []
    for key in "ABCDEFGHIJKLMNOPQRSTUVWXYZ 123456789":
        if win32api.GetAsyncKeyState(ord(key)):
            keys.append(key)
    return keys

def keys_to_output(output):
    # Convert keys to a multi-hot array [W,A,S,D]
    if 'W' in keys:
        output[0] = 1
    if 'A' in keys:
        output[1] = 1
    if 'S' in keys:
        output[2] = 1
    if 'D' in keys:
        output[3] = 1
    return output

np.set_printoptions(precision=3)
start_time = 0
output = np.zeros(OUTPUT_LENGTH)
new_data = np.empty((1000,INPUT_HEIGHT,INPUT_WIDTH),dtype='uint8')
new_data_counter = 0
forwards = np.empty((1000,INPUT_HEIGHT,INPUT_WIDTH),dtype='uint8')
forwards_counter = 0
left_right_brakes = np.empty((1000,INPUT_HEIGHT,INPUT_WIDTH),dtype='uint8')
correcting = False
paused = True
print('Press {} to unpause'.format(PAUSE_KEY))

model_exists = os.path.isfile('{}.index'.format(MODEL_NAME))
if model_exists:
    print('Loading previously trained model')
    model.load(MODEL_NAME)

previous_data_exists = os.path.isfile(DATA_FILE_NAME)
if previous_data_exists:
    print('Loading previously collected data')
    training_data = np.load(DATA_FILE_NAME)

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
            print('Training data at {} frames'.format(len(training_data)))
            print('Saving correction data')
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
        if correcting:
            correcting = False
            # print('Back under AI control')

    if not paused:
        start_time = time.time()

        frame = get_gta_window()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # resize to something a bit more acceptable for a CNN
        frame = cv2.resize(frame, (RESIZE_WIDTH, RESIZE_HEIGHT))
        frame = np.concatenate((frame, recent_keys))
        # frame.shape == (RESIZE_HEIGHT + OUTPUT_LENGTH, RESIZE_WIDTH)

        cv2.imshow('ALANN', cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation=cv2.INTER_NEAREST))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

        if model_exists:
            output = model.predict([frame.reshape(INPUT_WIDTH, INPUT_HEIGHT, 1)])[0]
            # print(output)
        else:
            output = np.zeros(OUTPUT_LENGTH)

        # If the prediction is saying the opposite of the correction then we
        # need to undo the prediction in addition to applying the correction
        if CORRECTING_KEYS[0] in keys:
            if output[1] or output[4] or output[6]: # if a or wa or sa
                output = wa
            elif output[3] or output[5] or output[7]: # if d or wd or sd
                output = wd
            else:
                output = w

        if CORRECTING_KEYS[1] in keys:
            if output[0] or output[4]: # if w or wa
                output = wa
            elif output[3] or output[5]: # if d or wd
                output = wa
            elif output[6] or output[7]: # if sa or sd
                output = sa
            else:
                output = a

        if CORRECTING_KEYS[2] in keys:
            if output[1] or output[4] or output[6]: # if a or wa or sa
                output = sa
            elif output[3] or output[5] or output[7]: # if d or wd or sd
                output = sd
            else:
                output = s

        if CORRECTING_KEYS[3] in keys:
            if output[0] or output[4]:  # if w or wa
                output = wd
            elif output[3] or output[5]:  # if d or wd
                output = wd
            elif output[6] or output[7]:  # if sa or sd
                output = sd
            else:
                output = d

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
            # Some pixels in the bottom right are used to store the key states
            # for the current frame in the training data
            frame[-OUTPUT_LENGTH:,-1] = output

            if np.array_equal(output,w):
                forwards[forwards_counter] = frame
                forwards_counter += 1
                forwards_counter %= 1000
            else:
                new_data[new_data_counter] = frame
                new_data_counter += 1
                if (forwards_counter % 1000) and not (new_data_counter == 1000):
                    new_data[new_data_counter] = np.random.choice(forwards[:forwards_counter])
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

        output_index = np.argmax(output)
        output = np.zeros(OUTPUT_LENGTH,dtype='uint8')
        output[output_index] = 1

        # Shift all values to the right
        recent_keys = np.roll(recent_keys, 1, axis=1)
        # Add the key states from this frame to the first column
        for i in range(0, OUTPUT_LENGTH):
            recent_keys[-i, 0] = output[i] * 255

            # During training these pixels in the bottom right are always black
            # because the key states are stored there. So they should also be black
            # when we run the model.
            frame[i - OUTPUT_LENGTH, -1] = 0

        duration = time.time() - start_time
        time.sleep(max(0, 1/18 - duration))

