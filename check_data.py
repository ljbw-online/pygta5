import numpy as np
import cv2
from time import time, sleep

from common import INITIAL_DATA_FILE_NAME, DISPLAY_WIDTH, DISPLAY_HEIGHT, OUTPUT_LENGTH

initial_data = np.load(INITIAL_DATA_FILE_NAME)

for frame in initial_data:
    start_time = time()
    cv2.imshow('initial data', cv2.resize(frame[:-1], (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation=cv2.INTER_NEAREST))
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

    outputs = list(frame[-1].reshape((-1, 4)))
    for i in range(len(outputs)):
        a = np.argmax(outputs[i])
        if a == 0:
            outputs[i] = 'w'
        elif a == 1:
            outputs[i] = 'wa'
        elif a == 2:
            outputs[i] = 'wd'
        elif a == 3:
            outputs[i] = 's'

    print(outputs)
    sleep(max(0., 1/5 - (time() - start_time)))
