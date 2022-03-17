import numpy as np
import cv2
from time import time, sleep

from common import (PAUSE_KEY, SAVE_AND_QUIT_KEY, INITIAL_DATA_FILE_NAME,
                    release_keys, QUIT_WITHOUT_SAVING_KEY, DISPLAY_WIDTH,
                    DISPLAY_HEIGHT, get_keys,
                    SAVE_AND_CONTINUE_KEY)

from multihot_3 import (
    create_datum, NUM_SIGNALS, DATUM_SHAPE,
    output_counts, save_datum_decision, get_frame, correction_keys_to_output,
    stop_session_decision, correction_to_keypresses)

# np.set_printoptions(precision=3)
start_time = 0

MAX_DATA_PER_OUTPUT = 5000
DATA_TARGET = MAX_DATA_PER_OUTPUT * NUM_SIGNALS

INITIAL_DATA_SHAPE = (DATA_TARGET,) + DATUM_SHAPE
initial_data = np.zeros(INITIAL_DATA_SHAPE, dtype='uint8')
print('initial_data.shape', initial_data.shape)

initial_data_index = 0


def display_frame(frame_param):
    frame_param = cv2.resize(frame_param, (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation=cv2.INTER_NEAREST)
    text_top_left = (round(DISPLAY_WIDTH * 0.1), round(DISPLAY_HEIGHT * 0.9))
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame_param, 'Collecting data', text_top_left, font, 2, (255, 255, 255), 3)
    cv2.imshow('ALANN', frame_param)
    # waitKey has to be called between imshow calls
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        return


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
            print('initial_data_index', initial_data_index)
            print('output_counts', output_counts)
            sleep(1)
    elif SAVE_AND_CONTINUE_KEY in keys:
        print('Saving {} frames'.format(initial_data_index + 1))
        initial_data = initial_data[:initial_data_index]
        np.save(INITIAL_DATA_FILE_NAME, initial_data)
        sleep(1)
    elif SAVE_AND_QUIT_KEY in keys:
        print('Saving {} frames'.format(initial_data_index + 1))
        initial_data = initial_data[:initial_data_index]
        np.save(INITIAL_DATA_FILE_NAME, initial_data)
        cv2.destroyAllWindows()
        release_keys()
        break
    elif QUIT_WITHOUT_SAVING_KEY in keys:
        print('Not saving correction data')
        cv2.destroyAllWindows()
        release_keys()
        break
    elif stop_session_decision(initial_data_index, DATA_TARGET):
        release_keys()
        paused = True
        print('{} frames collected.'.format(initial_data_index + 1))
        print('output_counts ==', output_counts)
        choice = input('Save? (y/n)\n')
        if choice == 'y':
            print('Saving {} frames to {}'.format(initial_data_index + 1, INITIAL_DATA_FILE_NAME))
            initial_data = initial_data[:initial_data_index]
            np.save(INITIAL_DATA_FILE_NAME, initial_data)
            cv2.destroyAllWindows()
            break
        elif choice == 'n':
            print('Not saving')
            cv2.destroyAllWindows()
            break

    if not paused:
        start_time = time()

        frame = get_frame()
        output = correction_keys_to_output(keys)
        datum = create_datum(frame, output)

        save_decision, output_counts, initial_data_index = save_datum_decision(
            keys, output_counts, initial_data_index, MAX_DATA_PER_OUTPUT)

        if save_decision:
            initial_data[initial_data_index] = datum

            if (initial_data_index % round(MAX_DATA_PER_OUTPUT/10) == 0) and (initial_data_index != 0):
                print(output_counts)

        correction_to_keypresses(keys)

        display_frame(frame)

        # duration = time() - start_time
        # sleep(max(0., round(1/18 - duration)))
