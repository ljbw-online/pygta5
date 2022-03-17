import numpy as np
import cv2
from time import sleep

from common import INITIAL_DATA_FILE_NAME, DISPLAY_WIDTH, DISPLAY_HEIGHT


def check(data):
    inner_break = False
    for datum in data:
        for frame in np.flip(datum, axis=0):
            cv2.imshow('initial data',
                       cv2.resize(frame[:-1], (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation=cv2.INTER_NEAREST))
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                inner_break = True
                break
            sleep(1/4)

        if inner_break:
            break

        output = datum[0, -1, 0:2].astype('bool')

        if output[0, 0]:
            fcb = 'forward'
        elif output[0, 1]:
            fcb = '       '
        elif output[0, 2]:
            fcb = 'brake  '
        else:
            fcb = 'huh?'

        if output[1, 0]:
            lsr = 'left'
        elif output[1, 1]:
            lsr = ''
        elif output[1, 2]:
            lsr = 'right'
        else:
            lsr = 'huh?'

        print(fcb, lsr)
        sleep(1)


if __name__ == '__main__':
    initial_data = np.load(INITIAL_DATA_FILE_NAME)
    check(initial_data)
