import numpy as np
import cv2
from time import time, sleep

from common import BALANCED_FILE_NAME, DISPLAY_WIDTH, DISPLAY_HEIGHT, OUTPUT_LENGTH

balanced_data = np.load(BALANCED_FILE_NAME)

if __name__ == '__main__':
    for frame in balanced_data:
        start_time = time()
        cv2.imshow('ALANN', cv2.resize(frame[:-1], (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation=cv2.INTER_NEAREST))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        print(['w','a','s','d','wa','wd','sa','sd','nk'][np.argmax(frame[-1,:OUTPUT_LENGTH])])
        sleep(max(0,1/5 - (time() - start_time)))