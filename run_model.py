from time import time, sleep
import multiprocessing as mp

import cv2

from common import PAUSE_KEY, release_keys, QUIT_WITHOUT_SAVING_KEY, get_keys, CORRECTION_KEYS
# from key_press_frames_to_multihot_4 import correction_to_keypresses
from actor_critic.run import ModelRunner

whole_loop_duration = 1

if __name__ == '__main__':
    q = mp.Queue()
    # p = mp.Process(target=display_features, args=(q,))
    # p.start()

    model_runner = ModelRunner(q)

    correcting = False
    paused = True
    get_keys()  # Flush key presses
    print('Press {} to unpause'.format(PAUSE_KEY))

    while True:
        start_time = time()
        keys = get_keys()

        if PAUSE_KEY in keys:
            if paused:
                paused = False
                print('Unpaused')
            else:
                paused = True
                release_keys()
                print('Paused')
            model_runner.pause_or_unpause(paused)
            sleep(1)
        elif set(CORRECTION_KEYS) & set(keys):
            if not correcting and not paused:
                correcting = True
        elif QUIT_WITHOUT_SAVING_KEY in keys:
            print('Quitting')
            model_runner.quit_model()
            cv2.destroyAllWindows()
            release_keys()
            # p.join()
            break
        else:
            if correcting:
                correcting = False

        paused = model_runner.paused

        if not paused:

            if correcting:
                pass
                # correction_to_keypresses(keys)

            model_runner.run_model(keys, correcting)

            while True:
                duration = time() - start_time
                if (start_time + 1/30) - time() > 0.015:  # 30fps
                    sleep(0.001)  # ~15ms
                else:
                    break

            # print(1 / (time() - start_time))
