import cv2
from time import sleep  # , time

from common import (PAUSE_KEY, SAVE_AND_QUIT_KEY,
                    release_keys, QUIT_WITHOUT_SAVING_KEY,
                    get_keys, SAVE_AND_CONTINUE_KEY, CORRECTING_KEYS)

from multihot_3 import (correction_to_keypresses, DataCollector, ModelRunner, NUM_SIGNALS)

# np.set_printoptions(precision=3)
# start_time = 0

MAX_DATA_PER_OUTPUT = 2000
DATA_TARGET = MAX_DATA_PER_OUTPUT * NUM_SIGNALS

data_collector = DataCollector(DATA_TARGET)
model_runner = ModelRunner()

data_collector.correction_data = True
model_runner.correction_data = True

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
            # print('self.index', data_collector.index)
            # print('Signals:', data_collector.signal_counts.list())
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
        # print('{} frames collected.'.format(data_collector.index))
        # print('Signals:', data_collector.signal_counts.list())
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
    elif set(CORRECTING_KEYS) & set(keys):
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
