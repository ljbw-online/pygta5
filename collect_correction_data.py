import numpy as np
import cv2
from time import time, sleep
from tensorflow.keras.models import load_model

from common import (INPUT_WIDTH, INPUT_HEIGHT, MODEL_NAME, get_gta_window, PAUSE_KEY, QUIT_WITHOUT_SAVING_KEY, PressKey,
                    W, A, S, D, key_check, CORRECTING_KEYS, DISPLAY_WIDTH, DISPLAY_HEIGHT, release_keys, OUTPUT_LENGTH,
                    QUIT_AND_SAVE_KEY, CORRECTION_FILE_NAME)

model = load_model(MODEL_NAME)
last_loop_time = 0
last_correcting_time = 0
paused = True
correcting = False
output = np.zeros(4, dtype='bool')
output_row = np.zeros((1, INPUT_WIDTH), dtype='float32')
correction_data = np.zeros((40000, INPUT_HEIGHT + 1, INPUT_WIDTH), dtype='uint8')
cd_buffer_count = 0
f_count = 0
fl_count = 0
fr_count = 0
b_count = 0

key_check()  # flush keys
print('Press {} to unpause'.format(PAUSE_KEY))
while True:
    keys = key_check()

    if PAUSE_KEY in keys:
        if paused:
            paused = False
            sleep(1)
        else:
            paused = True
            print('Correction data at {} frames'.format(cd_buffer_count))
            release_keys()
            sleep(1)
    elif set(CORRECTING_KEYS) & set(keys):
        if not correcting and not paused:
            correcting = True
    elif QUIT_AND_SAVE_KEY in keys:
        print('Saving {} frames'.format(cd_buffer_count))
        correction_data = correction_data[:cd_buffer_count]
        np.save(CORRECTION_FILE_NAME, correction_data)
        cv2.destroyAllWindows()
        release_keys()
        break
    elif QUIT_WITHOUT_SAVING_KEY in keys:
        release_keys()
        cv2.destroyAllWindows()
        break
    elif cd_buffer_count == 39999:
        print('40k frames recorded')
        print('Press {} to save and quit'.format(QUIT_AND_SAVE_KEY))
        release_keys()
    else:
        if correcting:
            correcting = False

    if not paused:
        last_loop_time = time()

        frame = get_gta_window()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (INPUT_WIDTH, INPUT_HEIGHT))
        frame = frame.astype('float32') / 255.0

        prediction = model.predict(frame.reshape(1, INPUT_HEIGHT, INPUT_WIDTH, 1))[0]

        output[:] = False
        if correcting:
            last_correcting_time = time()

            if CORRECTING_KEYS[2] in keys:  # If I'm pressing s then brake regardless of other keys
                output[3] = True
            elif CORRECTING_KEYS[1] in keys:
                output[1] = True
            elif CORRECTING_KEYS[3] in keys:
                output[2] = True
            elif CORRECTING_KEYS[0] in keys:  # Only press forward if I am not pressing left or right
                output[0] = True

            output_row[0, :OUTPUT_LENGTH] = output.astype('float32')
            if output[0] and (f_count < fl_count):
                correction_data[cd_buffer_count] = np.concatenate((frame, output_row))
                cd_buffer_count += 1
                f_count += 1
            elif output[1] and (fl_count < fr_count):
                correction_data[cd_buffer_count] = np.concatenate((frame, output_row))
                cd_buffer_count += 1
                fl_count += 1
            elif output[3] and (fr_count - fl_count < 100):  # There can be up to 100 more fr's than fl's
                correction_data[cd_buffer_count] = np.concatenate((frame, output_row))
                cd_buffer_count += 1
                fr_count += 1
            elif output[2] and (b_count < f_count):
                correction_data[cd_buffer_count] = np.concatenate((frame, output_row))
                cd_buffer_count += 1
                b_count += 1

        else:
            argmax = np.argmax(prediction)
            output[argmax] = True

        display_frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation=cv2.INTER_NEAREST)
        if (time() - last_correcting_time) < 1:
            text_top_left = (round(DISPLAY_WIDTH*0.1), round(DISPLAY_HEIGHT*0.9))
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(display_frame, 'Under human control', text_top_left, font, 2, (255, 255, 255), 3)

        cv2.imshow('ALANN', display_frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

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

        duration = time() - last_loop_time
        sleep(max(0, round(1/18 - duration)))
