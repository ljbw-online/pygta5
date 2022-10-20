import multiprocessing as mp
# import queue
from time import sleep, time

import win32api
import numpy as np
import cv2


def foo(q_param):
    while True:
        # print('main process says: ', q_param.get())
        # sleep(1)
        # sub_t = time()

        while True:
            array, num = q_param.get()

            # print('qsize', q_param.qsize())

            if q_param.qsize() == 0:  # q_param.empty() returns True even when the queue isn't empty...
                # print('queue empty', num)
                break
            else:
                # print('not empty', num)
                pass

        # while True:
        #     try:
        #         array, num = q_param.get_nowait()
        #     except queue.Empty:
        #         break

        # print(type(array))

        if type(array) is np.ndarray:
            print('imshowing', num)
            cv2.imshow('n', array)
            cv2.waitKey(1)
        else:
            # print('destroying windows', num)
            # cv2.destroyAllWindows()
            break

        # print(round(1/(time() - t)))
        sleep(0.05)

        # print('sub', round(1. / (time() - sub_t)))


if __name__ == '__main__':
    q = mp.Queue()
    # q.put('hello world')
    p = mp.Process(target=foo, args=(q,))
    p.start()

    font = cv2.FONT_HERSHEY_SIMPLEX

    n = 0
    while True:
        # t = time()
        print('loop', str(n))

        a = np.zeros((500, 500), dtype='uint8')
        a = cv2.putText(a, str(n), (200, 200), font, 2, (255,), 3)

        q.put((a, n))

        n += 1

        if win32api.GetAsyncKeyState(ord('A')):
            print('A pressed. Goodbye.')
            q.put((None, n))
            break

        # sleep(0.016667)
        # sleep(max(0., 1/60 - (time() - t)))
        sleep(0.005)  # this loop doesn't go above 60fps when I call sleep for some reason. Without sleep it does about
        # 1000fps.

        # print('main', round(1. / (time() - t)))

    # sleep(1)
    cv2.destroyAllWindows()
    p.join()
