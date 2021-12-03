import numpy as np
# from pandas import DataFrame
import cv2
# from collections import Counter
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
    lefts = 0
    brakes = 0
    rights = 0
    forward_lefts = 0
    forward_rights = 0
    brake_lefts = 0
    brake_rights = 0
    nokeys = 0

    for keystate in training_data[:,-1,:OUTPUT_LENGTH]:
        if np.array_equal(keystate,w):
            forwards += 1
        elif np.array_equal(keystate,a):
            lefts += 1
        elif np.array_equal(keystate,s):
            brakes += 1
        elif np.array_equal(keystate,d):
            rights += 1
        elif np.array_equal(keystate,wa):
            forward_lefts += 1
        elif np.array_equal(keystate,wd):
            forward_rights += 1
        elif np.array_equal(keystate,sa):
            brake_lefts += 1
        elif np.array_equal(keystate,sd):
            brake_rights += 1
        elif np.array_equal(keystate,nk):
            nokeys += 1
        else:
            print('huh?',keystate)

    print('forwards',forwards)
    print('lefts',lefts)
    print('brakes',brakes)
    print('rights',rights)
    print('forward_lefts',forward_lefts)
    print('forward_rights',forward_rights)
    print('brake_lefts',brake_lefts)
    print('brake_rights',brake_rights)
    print('nokeys',nokeys)

    for frame in training_data:
        start_time = time()
        cv2.imshow('ALANN', cv2.resize(frame[:-1], (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation=cv2.INTER_NEAREST))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        print(['w','a','s','d','wa','wd','sa','sd','nk'][np.argmax(frame[-1,:OUTPUT_LENGTH])])
        sleep(max(0,1/5 - (time() - start_time)))