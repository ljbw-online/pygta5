import numpy as np
import cv2
from time import time, sleep
from tensorflow.keras.models import load_model

from common import (INPUT_WIDTH, INPUT_HEIGHT, get_gta_window,
                    PAUSE_KEY, SAVE_AND_QUIT_KEY, CORRECTION_DATA_FILE_NAME,
                    PressKey, release_keys, CORRECTING_KEYS,
                    QUIT_WITHOUT_SAVING_KEY, OUTPUT_LENGTH, DISPLAY_WIDTH,
                    DISPLAY_HEIGHT, RESIZE_WIDTH, RESIZE_HEIGHT, key_check,
                    W, A, S, D, SAVE_AND_CONTINUE_KEY, MODEL_NAME)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

np.set_printoptions(precision=3)
FRAMES_PER_OUTPUT = 2000  # A lower value in this file because collecting correction data is much slower

model = load_model(MODEL_NAME)
start_time = 0
last_correcting_time = 0
output = np.zeros(OUTPUT_LENGTH, dtype='bool')
output_row = np.zeros((1, INPUT_WIDTH), dtype='uint8')
correction_data_index = 0
forwards = np.zeros((FRAMES_PER_OUTPUT, INPUT_HEIGHT + 1, INPUT_WIDTH), dtype='uint8')
f_buffer_index = 0
f_count = 0
lefts = forwards.copy()
l_buffer_index = 0
l_count = 0
rights = forwards.copy()
r_buffer_index = 0
r_count = 0
brakes = forwards.copy()
b_buffer_index = 0
b_count = 0

paused = True
correcting = False
print('Press {} to unpause'.format(PAUSE_KEY))
key_check()  # Flush key presses

while True:
    keys = key_check()

    if PAUSE_KEY in keys:
        if paused:
            paused = False
            print('Unpaused')
            sleep(1)
        else:
            release_keys()
            paused = True
            print('Paused')
            print('initial_data_index', correction_data_index)
            print(f_buffer_index, l_buffer_index, r_buffer_index, b_buffer_index)
            sleep(1)
    elif SAVE_AND_CONTINUE_KEY in keys:
        correction_data_uint8 = np.concatenate((forwards, lefts, rights, brakes))
        print('Saving {} frames'.format(len(correction_data_uint8)))
        np.save(CORRECTION_DATA_FILE_NAME, correction_data_uint8)
        sleep(1)
    elif SAVE_AND_QUIT_KEY in keys:
        correction_data_uint8 = np.concatenate((forwards, lefts, rights, brakes))
        print('Saving {} frames'.format(len(correction_data_uint8)))
        np.save(CORRECTION_DATA_FILE_NAME, correction_data_uint8)
        cv2.destroyAllWindows()
        release_keys()
        break
    elif QUIT_WITHOUT_SAVING_KEY in keys:
        print('Not saving correction data')
        cv2.destroyAllWindows()
        release_keys()
        break
    elif correction_data_index == FRAMES_PER_OUTPUT * OUTPUT_LENGTH:
        release_keys()
        paused = True
        choice = input('{} frames collected. Save? (y/n)\n'.format(FRAMES_PER_OUTPUT * OUTPUT_LENGTH))
        if choice == 'y':
            print('Saving {} frames to {}'.format(correction_data_index, CORRECTION_DATA_FILE_NAME))
            correction_data_uint8 = np.concatenate((forwards, lefts, rights, brakes))
            np.save(CORRECTION_DATA_FILE_NAME, correction_data_uint8)
            cv2.destroyAllWindows()
            break
        elif choice == 'n':
            print('Not saving')
            cv2.destroyAllWindows()
            break
        else:
            print('huh?')

    if not paused:
        start_time = time()

        frame = get_gta_window()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (RESIZE_WIDTH, RESIZE_HEIGHT))
        # frame.shape == (RESIZE_HEIGHT, RESIZE_WIDTH)

        output[:] = False
        if CORRECTING_KEYS[2] in keys:
            output[3] = True
            correcting = True
        elif CORRECTING_KEYS[1] in keys:
            output[1] = True
            correcting = True
        elif CORRECTING_KEYS[3] in keys:
            output[2] = True
            correcting = True
        elif CORRECTING_KEYS[0] in keys:
            output[0] = True
            correcting = True
        else:
            correcting = False
            image_input = frame.reshape(1, INPUT_HEIGHT, INPUT_WIDTH, 1).astype('float32') / 255.0
            # keys_input = output_row[0, :-OUTPUT_LENGTH].reshape(1, -1).astype('float32')
            # , 'keys_input': keys_input
            prediction = model({'image_input': image_input}).numpy()
            argmax = np.argmax(prediction)
            output[argmax] = True

        output_row = np.roll(output_row, OUTPUT_LENGTH)
        output_row[0, :OUTPUT_LENGTH] = output.astype('uint8')

        if correcting:
            last_correcting_time = time()
            datum = np.concatenate((frame, output_row))

            # The number of forwards frames cannot exceed that of any of the other output types. This ensures that
            # forwards frames are collected gradually throughout the session.
            if (output[0] and f_buffer_index < min([l_buffer_index, r_buffer_index, b_buffer_index])
                    and not f_buffer_index == FRAMES_PER_OUTPUT):
                forwards[f_buffer_index] = datum
                f_buffer_index += 1
            elif output[1] and not l_buffer_index == FRAMES_PER_OUTPUT:
                lefts[l_buffer_index] = datum
                l_buffer_index += 1
            elif output[2] and not r_buffer_index == FRAMES_PER_OUTPUT:
                rights[r_buffer_index] = datum
                r_buffer_index += 1
            elif output[3] and not b_buffer_index == FRAMES_PER_OUTPUT:
                brakes[b_buffer_index] = datum
                b_buffer_index += 1

            correction_data_index = f_buffer_index + l_buffer_index + r_buffer_index + b_buffer_index

            if correction_data_index % 1000 == 0:
                print(f_buffer_index, l_buffer_index, r_buffer_index, b_buffer_index)

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

        frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation=cv2.INTER_NEAREST)
        if (time() - last_correcting_time) < 1:
            text_top_left = (round(DISPLAY_WIDTH*0.1), round(DISPLAY_HEIGHT*0.9))
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, 'Correcting', text_top_left, font, 2, (255, 255, 255), 3)

        cv2.imshow('ALANN', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

        # duration = time() - start_time
        # print(1./(time() - start_time))
        # sleep(max(0., round(1/18 - duration)))
