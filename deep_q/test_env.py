import cv2
import numpy as np

from common import get_keys, resize
from deep_q.breakout_wrapper import Env, num_actions, apply_correction


def main():
    env = Env()

    get_keys()
    while True:
        state = env.reset()
        while True:
            keys = get_keys()

            try:
                action = min(int(''.join(keys)), num_actions - 1)
            except ValueError:
                action = 0
                # action = np.random.choice(num_actions)

            action, _ = apply_correction(keys, action)

            state_next, reward, terminated = env.step(action, keys)
            # print(reward)

            cv2.imshow('test', resize(state, width=640))
            cv2.waitKey(16)

            state = state_next

            if 'Q' in keys or terminated:
                break

        if 'Q' in keys:
            cv2.destroyAllWindows()
            env.close()
            break


if __name__ == '__main__':
    main()
