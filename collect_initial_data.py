import numpy as np
from numpy import array_equal as ae
import cv2
from time import time, sleep

from common import (INPUT_WIDTH, INPUT_HEIGHT, get_gta_window,
                    PAUSE_KEY, QUIT_AND_SAVE_KEY, INITIAL_DATA_FILE_NAME,
                    PressKey, release_keys, CORRECTING_KEYS,
                    QUIT_WITHOUT_SAVING_KEY, OUTPUT_LENGTH, DISPLAY_WIDTH,
                    DISPLAY_HEIGHT, RESIZE_WIDTH, RESIZE_HEIGHT, key_check,
                    W, A, S, D)

np.set_printoptions(precision=3)

start_time = 0
last_correcting_time = 0
output = np.zeros(OUTPUT_LENGTH, dtype='bool')
output_row = np.zeros((1,INPUT_WIDTH), dtype='uint8')
initial_data_uint8 = np.empty((40000, INPUT_HEIGHT + 1, INPUT_WIDTH), dtype='uint8')
initial_data_index = 0
forwards = np.zeros((1000, INPUT_HEIGHT + 1, INPUT_WIDTH), dtype='uint8')
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
            sleep(1)
            paused = True
            print('Paused')
            print('new_data_counter', initial_data_index)
            print(f_buffer_index, l_buffer_index, r_buffer_index, b_buffer_index)
    elif QUIT_AND_SAVE_KEY in keys:
        training_data = initial_data_uint8[:initial_data_index]
        print('Saving {} frames'.format(len(training_data)))
        np.save(INITIAL_DATA_FILE_NAME, training_data)
        cv2.destroyAllWindows()
        release_keys()
        break
    elif QUIT_WITHOUT_SAVING_KEY in keys:
        print('Not saving correction data')
        cv2.destroyAllWindows()
        release_keys()
        break
    elif initial_data_index == 40000:
        paused = True
        choice = input('40k frames collected. Save? (y/n)\n')
        if choice == 'y':
            print('Saving {} frames to {}'.format(initial_data_index, INITIAL_DATA_FILE_NAME))
            np.save(INITIAL_DATA_FILE_NAME, initial_data_uint8)
            cv2.destroyAllWindows()
            release_keys()
            break
        elif choice == 'n':
            print('Not saving')
            cv2.destroyAllWindows()
            release_keys()
            break
        else:
            print('huh?')

    if not paused:
        start_time = time()

        frame = get_gta_window()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (RESIZE_WIDTH, RESIZE_HEIGHT))
        # frame.shape == (RESIZE_HEIGHT + OUTPUT_LENGTH, RESIZE_WIDTH)

        output[:] = False
        if CORRECTING_KEYS[2] in keys:
            output[3] = True
        elif CORRECTING_KEYS[1] in keys:
            output[1] = True
        elif CORRECTING_KEYS[3] in keys:
            output[2] = True
        else:  # By default press forward
            output[0] = True

        output_row = np.roll(output_row, OUTPUT_LENGTH)
        output_row[0, :OUTPUT_LENGTH] = output.astype('uint8')
        # print(list(map(np.argmax, output_row.reshape((-1, 4)))))

        datum = np.concatenate((frame, output_row))
        # If we have 1000 of any output then stop saving it to it's buffer
        if output[0] and not f_buffer_index == 1000:
            forwards[f_buffer_index] = datum
            f_buffer_index += 1
        elif output[1] and not l_buffer_index == 1000:
            lefts[l_buffer_index] = datum
            l_buffer_index += 1
        elif output[2] and not r_buffer_index == 1000:
            rights[r_buffer_index] = datum
            r_buffer_index += 1
        elif output[3] and not b_buffer_index == 1000:
            brakes[b_buffer_index] = datum
            b_buffer_index += 1

        count_list = [f_count, l_count, r_count, b_count]
        current_count = count_list[np.argmax(output)]  # argmax works for bools
        max_count = max(count_list)

        # Once we have 1000 of each output save them all to initial_data_uint8
        if f_buffer_index == 1000 and l_buffer_index == 1000 and r_buffer_index == 1000 and b_buffer_index == 1000:

            data = np.concatenate((forwards, lefts, rights, brakes))

            initial_data_uint8[initial_data_index:(initial_data_index + 4000)] = data
            initial_data_index += 4000

            f_buffer_index = 0
            l_buffer_index = 0
            r_buffer_index = 0
            b_buffer_index = 0

            print('Data at {} frames'.format(initial_data_index))

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
        text_top_left = (round(DISPLAY_WIDTH*0.1), round(DISPLAY_HEIGHT*0.9))
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, 'Under human control', text_top_left, font, 2, (255, 255, 255), 3)

        cv2.imshow('ALANN', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

        duration = time() - start_time
        sleep(max(0, round(1/18 - duration)))
