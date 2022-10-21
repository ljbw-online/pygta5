from time import sleep  # , time
import multiprocessing as mp

import cv2

from common import (PAUSE_KEY, SAVE_AND_QUIT_KEY,
                    release_keys, QUIT_WITHOUT_SAVING_KEY,
                    get_keys, SAVE_AND_CONTINUE_KEY, CORRECTION_KEYS)

from key_press_frames_to_multihot_4 import correction_to_keypresses
from key_press_frames_to_multihot_4.run import ModelRunner
from key_press_frames_to_multihot_4.collect_correction import DataCollector

q = mp.Queue

data_collector = DataCollector(50000)
model_runner = ModelRunner(q)

# data_collector.correction_data = True
# model_runner.correction_data = True

paused = True
correcting = False
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
            data_collector.print_signals()
            sleep(1)
    elif SAVE_AND_CONTINUE_KEY in keys:
        print('NOT IMPLEMENTED')
        sleep(1)
    elif SAVE_AND_QUIT_KEY in keys:
        data_collector.save()
        cv2.destroyAllWindows()
        release_keys()
        break
    elif QUIT_WITHOUT_SAVING_KEY in keys:
        print('Not saving correction data')
        cv2.destroyAllWindows()
        release_keys()
        break
    elif data_collector.stop_session_decision():
        release_keys()
        paused = True
        data_collector.print_signals()
        choice = input('Save? (y/n)\n')
        if choice == 'y':
            data_collector.save()
            cv2.destroyAllWindows()
            break
        elif choice == 'n':
            print('Not saving')
            cv2.destroyAllWindows()
            break
    # This has to be below the stop_session_decision branch otherwise stop_session_decision would only be evaluated
    # when I release the correction keys. This can cause correction_data_index to be equal to DATA_TARGET which then
    # causes IndexError.
    elif set(CORRECTION_KEYS) & set(keys):
        if not paused:
            correcting = True
    else:
        correcting = False

    if not paused:
        ##############
        if correcting:
            correction_to_keypresses(keys)
            data_collector.collect_datum(keys)

        model_runner.run_model(keys, correcting)

        get_keys()  # flush GetAsyncKeyState buffer
