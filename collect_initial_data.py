from time import sleep, time
import multiprocessing as mp

import cv2

from common import (PAUSE_KEY, SAVE_AND_QUIT_KEY,
                    release_keys, QUIT_WITHOUT_SAVING_KEY, get_keys,
                    SAVE_AND_CONTINUE_KEY)

from frame_seq_to_mh4 import correction_to_keypresses, DataCollector, display_frame

duration = 1
sleep_duration = 0
whole_loop_duration = 1

if __name__ == '__main__':
    q = mp.Queue()  # q.put doesn't block whereas conn.send does, so have to use a queue
    p = mp.Process(target=display_frame, args=(q,))
    p.start()

    data_collector = DataCollector(10000, q)

    paused = True
    get_keys()  # Flush key presses
    t0 = time()
    print('Press {} to unpause'.format(PAUSE_KEY))

    while True:
        start_time = time()
        keys = get_keys()

        if PAUSE_KEY in keys:
            if paused:
                paused = False
                print('Unpaused')
                sleep(1)
            else:
                paused = True
                release_keys()
                data_collector.print_signals()
                print('Paused')
                sleep(1)
        elif SAVE_AND_CONTINUE_KEY in keys:
            print('NOT IMPLEMENTED')
            sleep(1)
        elif SAVE_AND_QUIT_KEY in keys:
            release_keys()
            data_collector.save()
            data_collector.quit_collector()
            cv2.destroyAllWindows()
            p.join()
            break
        elif QUIT_WITHOUT_SAVING_KEY in keys:
            print('Not saving correction data')
            release_keys()
            data_collector.quit_collector()
            cv2.destroyAllWindows()
            p.join()
            break
        elif data_collector.stop_session_decision():
            paused = True
            release_keys()
            data_collector.quit_collector()
            cv2.destroyAllWindows()

            data_collector.print_signals()
            choice = input('Save? (y/n)\n')
            if choice == 'y':
                data_collector.save()
                break
            elif choice == 'n':
                print('Not saving')
                break

            p.join()

        if not paused:
            # print(round(1. / whole_loop_duration), duration, sleep_duration)
            fps = round(1. / whole_loop_duration)
            if fps < 30:
                print(fps)
            elif fps > 40:
                print('   ', fps)
            else:
                print(' ', fps)

            data_collector.collect_datum(keys)

            correction_to_keypresses(keys)
            get_keys()  # This has to be here, otherwise W is always present in keys. PressKey and
            # ReleaseKey can be used to press and release W very rapidly so I don't know why this is necessary. get_key
            # again appears to be exhibiting some kind of flushing behaviour which refreshes the list of keys which are
            # detected.

            while True:
                duration = time() - start_time
                if duration < 1./40:
                    sleep(0.001)  # ~15ms
                else:
                    break
            # sleep_duration = max(0., 1. / 60 - duration)
            # print('sleeping for', sleep_duration, 'seconds')
            sleep(sleep_duration)
            whole_loop_duration = time() - start_time
