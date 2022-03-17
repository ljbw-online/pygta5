import numpy as np
import cv2
from time import time, sleep
import os

from common import INITIAL_DATA_FILE_NAME, DISPLAY_WIDTH, DISPLAY_HEIGHT, CORRECTION_DATA_FILE_NAME

from multihot_3 import output_row_to_wasd_string

data = None  # Stops "'data' can be undefined"
if os.path.isfile(CORRECTION_DATA_FILE_NAME):
    choice = input('Check correction data? (y/n)\n')
    if choice == 'y':
        print('Loading correction data')
        data = np.load(CORRECTION_DATA_FILE_NAME)
    else:
        print('Loading initial data')
        data = np.load(INITIAL_DATA_FILE_NAME)
        print(data.shape)
        exit()

for datum in data:
    start_time = time()
    frame = datum[:-1]
    output_row = datum[-1]
    cv2.imshow('data', cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation=cv2.INTER_NEAREST))
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

    print(output_row_to_wasd_string(output_row))

    # sleep(max(0., 1/5 - (time() - start_time)))

cv2.destroyAllWindows()
