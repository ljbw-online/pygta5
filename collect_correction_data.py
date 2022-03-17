import numpy as np
import cv2
from time import time, sleep
from tensorflow.keras.models import load_model, Model

from common import (PAUSE_KEY, SAVE_AND_QUIT_KEY,
                    release_keys, QUIT_WITHOUT_SAVING_KEY,
                    get_keys, SAVE_AND_CONTINUE_KEY, INPUT_WIDTH, INPUT_HEIGHT, CORRECTING_KEYS,
                    CORRECTION_DATA_FILE_NAME)

from multihot_3 import (
    create_datum, NUM_SIGNALS, DATUM_SHAPE,
    output_counts, save_datum_decision, get_frame, correction_keys_to_output,
    stop_session_decision, correction_to_keypresses, display_features, prediction_to_key_presses)

# np.set_printoptions(precision=3)
start_time = 0

# Things can't be imported from collect_initial_data because they'd use the MAX_DATA_PER_OUTPUT value from that file
# which is probably larger than what I want for this file. Also, to import anything from that module or
# run_model I would have to put their while-loops inside "if __name__ == '__main__'".
model = load_model('multihot_3')
model = Model(inputs=model.inputs, outputs=[layer.output for layer in model.layers])
prediction = []

MAX_DATA_PER_OUTPUT = 2500
DATA_TARGET = MAX_DATA_PER_OUTPUT * NUM_SIGNALS

DATA_SHAPE = (DATA_TARGET,) + DATUM_SHAPE
correction_data = np.zeros(DATA_SHAPE, dtype='uint8')
print('correction_data.shape', correction_data.shape)

correction_data_index = 0

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
            print('correction_data_index', correction_data_index)
            print('output_counts', output_counts)
            sleep(1)
    elif SAVE_AND_CONTINUE_KEY in keys:
        print('Saving {} frames'.format(correction_data_index + 1))
        correction_data_to_save = correction_data[:correction_data_index]
        np.save(CORRECTION_DATA_FILE_NAME, correction_data_to_save)
        sleep(1)
    elif SAVE_AND_QUIT_KEY in keys:
        print('Saving {} frames'.format(correction_data_index + 1))
        correction_data = correction_data[:correction_data_index]
        np.save(CORRECTION_DATA_FILE_NAME, correction_data)
        cv2.destroyAllWindows()
        release_keys()
        break
    elif QUIT_WITHOUT_SAVING_KEY in keys:
        print('Not saving correction data')
        cv2.destroyAllWindows()
        release_keys()
        break
    elif stop_session_decision(correction_data_index, DATA_TARGET):
        release_keys()
        paused = True
        print('{} frames collected.'.format(correction_data_index + 1))
        print('output_counts ==', output_counts)
        choice = input('Save? (y/n)\n')
        if choice == 'y':
            print('Saving {} frames to {}'.format(correction_data_index + 1, CORRECTION_DATA_FILE_NAME))
            correction_data = correction_data[:correction_data_index]
            np.save(CORRECTION_DATA_FILE_NAME, correction_data)
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
        start_time = time()

        frame = get_frame()

        if correcting:
            correction_to_keypresses(keys)

            output = correction_keys_to_output(keys)
            datum = create_datum(frame, output)

            save_decision, output_counts, correction_data_index = save_datum_decision(
                keys, output_counts, correction_data_index, MAX_DATA_PER_OUTPUT)

            if save_decision:
                correction_data[correction_data_index] = datum

                if (correction_data_index % round(MAX_DATA_PER_OUTPUT / 10) == 0) and (correction_data_index != 0):
                    print(output_counts)
        else:
            frame = frame.reshape(1, INPUT_HEIGHT, INPUT_WIDTH, 3)
            prediction = model(frame)  # model returns a list of tensors corresponding to each layer
            prediction_to_key_presses(prediction[-1][0])

        display_features(prediction, correcting=correcting)

        # duration = time() - start_time
        # sleep(max(0., round(1/18 - duration)))
