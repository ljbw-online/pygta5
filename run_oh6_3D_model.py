import numpy as np
import cv2
from time import time, sleep
from tensorflow.keras.models import load_model

from common import (INPUT_WIDTH, INPUT_HEIGHT, MODEL_NAME, get_gta_window, PAUSE_KEY, QUIT_WITHOUT_SAVING_KEY, PressKey,
                    W, A, S, D, get_keys, CORRECTING_KEYS, DISPLAY_WIDTH, DISPLAY_HEIGHT, release_keys, OUTPUT_SHAPE,
                    FRAME_SEQ_LENGTH, FRAME_BUFFER_LENGTH)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

model = load_model(MODEL_NAME)

output = np.zeros(OUTPUT_SHAPE, dtype='bool')
output_row = np.zeros((1, INPUT_WIDTH, 3), dtype='uint8')
datum = np.zeros((FRAME_SEQ_LENGTH, INPUT_HEIGHT + 1, INPUT_WIDTH, 3), dtype='uint8')
frame_buffer = np.zeros((FRAME_BUFFER_LENGTH, INPUT_HEIGHT + 1, INPUT_WIDTH, 3), dtype='uint8')
frame_time_buffer = np.arange(1/FRAME_BUFFER_LENGTH, 1 + 1/FRAME_BUFFER_LENGTH, 1/FRAME_BUFFER_LENGTH, dtype='float64')

last_loop_time = 0
last_correcting_time = 0
paused = True
correcting = False
get_keys()  # flush keys

print('Press {} to unpause'.format(PAUSE_KEY))
while True:
    keys = get_keys()

    if PAUSE_KEY in keys:
        if paused:
            paused = False
            sleep(1)
        else:
            paused = True
            release_keys()
            sleep(1)
    elif set(CORRECTING_KEYS) & set(keys):
        if not correcting and not paused:
            correcting = True
    elif QUIT_WITHOUT_SAVING_KEY in keys:
        release_keys()
        cv2.destroyAllWindows()
        break
    else:
        if correcting:
            correcting = False

    if not paused:
        last_loop_time = time()

        frame = get_gta_window()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        frame = cv2.resize(frame, (INPUT_WIDTH, INPUT_HEIGHT))
        frame = frame.astype('float32') / 255.0

        if correcting:
            last_correcting_time = time()

        display_frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation=cv2.INTER_NEAREST)
        if (time() - last_correcting_time) < 1:
            text_top_left = (round(DISPLAY_WIDTH*0.1), round(DISPLAY_HEIGHT*0.9))
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(display_frame, 'Under human control', text_top_left, font, 2, (255, 255, 255), 3)

        cv2.imshow('ALANN', display_frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

        output_row = np.roll(output_row, OUTPUT_SHAPE[0], axis=0)
        output_row[0, 0:2] = output.astype('uint8')

        frame_time_buffer = np.roll(frame_time_buffer, 1)
        frame_time_buffer[0] = last_loop_time
        frame_with_output_row = np.concatenate((frame, output_row))
        frame_buffer = np.roll(frame_buffer, 1, axis=0)
        frame_buffer[0] = frame_with_output_row
        # print(datum.shape, frame_with_output_row.shape)
        datum[0] = frame_with_output_row

        for i in range(len(frame_time_buffer) - 1):
            t_0 = last_loop_time - frame_time_buffer[i]
            t_1 = last_loop_time - frame_time_buffer[i+1]
            fslmo = FRAME_SEQ_LENGTH - 1
            # Take (FRAME_SEQ_LENGTH - 1) equally spaced frames from the last second of recording
            # This isn't the clearest way of writing this but it allows me to easily change FRAME_SEQ_LENGTH
            for j in range(1, FRAME_SEQ_LENGTH):
                if (t_0 <= j/fslmo) and (t_1 >= j/fslmo):
                    datum[j] = frame_buffer[i]

        if last_loop_time - frame_time_buffer[-1] <= 1.0:
            print('frame_buffer not long enough')

        output[:] = False
        if correcting:

            if CORRECTING_KEYS[0] in keys:
                output[0, 0] = True
            elif CORRECTING_KEYS[2] in keys:
                output[0, 2] = True
            else:
                output[0, 1] = True

            if CORRECTING_KEYS[1] in keys:
                output[1, 0] = True
            elif CORRECTING_KEYS[3] in keys:
                output[1, 2] = True
            else:
                output[1, 1] = True

        else:
            # The model doesn't take the last row of datum
            prediction = model(datum[:, :-1].reshape(1, FRAME_SEQ_LENGTH, INPUT_HEIGHT, INPUT_WIDTH, 3))

            fcb_argmax = np.argmax(prediction[0])
            output[0, fcb_argmax] = True

            lsr_argmax = np.argmax(prediction[1])
            output[1, lsr_argmax] = True

            print(prediction[0].numpy(), prediction[1].numpy())

        release_keys()
        if output[0, 0]:
            PressKey(W)
        elif output[0, 2]:
            PressKey(S)

        if output[1, 0]:
            PressKey(A)
        elif output[1, 2]:
            PressKey(D)

        # duration = time() - last_loop_time
        # sleep(max(0., 1/18 - duration))
