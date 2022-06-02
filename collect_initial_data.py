from time import sleep  # , time
import cv2

from common import (PAUSE_KEY, SAVE_AND_QUIT_KEY,
                    release_keys, QUIT_WITHOUT_SAVING_KEY, get_keys,
                    SAVE_AND_CONTINUE_KEY)

from key_press_frames_to_multihot_4 import (correction_to_keypresses, DataCollector)

# start_time = 0

# MAX_DATA_PER_OUTPUT = 10000
# DATA_TARGET = MAX_DATA_PER_OUTPUT * NUM_SIGNALS

data_collector = DataCollector(100)

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
            data_collector.print_signals()
            print('Paused')
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

    if not paused:
        # start_time = time()

        data_collector.collect_datum(keys)

        correction_to_keypresses(keys)
        get_keys()  # This has to be here, otherwise W is always present in keys. PressKey and
        # ReleaseKey can be used to press and release W very rapidly so I don't know why this is necessary. get_key
        # again appears to be exhibiting some kind of flushing behaviour which refreshes the list of keys which are
        # detected.

        # duration = time() - start_time
        # sleep(max(0., round(1/18 - duration)))
