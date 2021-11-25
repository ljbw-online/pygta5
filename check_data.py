import numpy as np
from pandas import DataFrame
import cv2
from collections import Counter
from random import shuffle
from time import time, sleep

from common import (DATA_FILE_NAME, DISPLAY_WIDTH, DISPLAY_HEIGHT, OUTPUT_LENGTH,
    w,a,s,d,wa,wd,sa,sd,nk)

if __name__ == '__main__':
    training_data = np.load(DATA_FILE_NAME)

    # df = DataFrame(training_data[:,-4:,-1],columns=['w','a','s','d'])
    # print(df.head())
    # print(Counter(df[1].apply(str)))
    # No longer useful due to new format for training_data

    forwards = 0
    brakes = 0
    lefts = 0
    rights = 0
    nokeys = 0

    for keystate in training_data[:,-OUTPUT_LENGTH:,-1]:
        if np.array_equal(keystate,w):
            forwards += 1
        elif np.array_equal(keystate,s):
            brakes += 1
        elif np.array_equal(keystate,wa) or np.array_equal(keystate,sa) or np.array_equal(keystate,a):
            lefts += 1
        elif np.array_equal(keystate,wd) or np.array_equal(keystate,sd) or np.array_equal(keystate,d):
            rights += 1
        else:
            nokeys += 1

    print('forwards',forwards)
    print('brakes',brakes)
    print('lefts',lefts)
    print('rights',rights)
    print('nokeys',nokeys)
    for frame in training_data:
        start_time = time()
        cv2.imshow('ALANN', cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation=cv2.INTER_NEAREST))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        sleep(max(0,1/5 - (time() - start_time)))