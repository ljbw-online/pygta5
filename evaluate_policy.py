import os

import keras
import numpy as np
import cv2

from common import model_dir
from dqn import env, compute_average_return, best_model_path

rng = np.random.default_rng()


if __name__ == '__main__':
    # single_step = True
    single_step = False
    # env = Env()

    if single_step:
        env.render()

    average_return = compute_average_return(env, keras.models.load_model(best_model_path), num_episodes=30,
                                            single_step=single_step, render_env=True)

    if average_return is not None:
        print(average_return)

        if single_step:
            cv2.destroyWindow(env.name)

    env.close()
