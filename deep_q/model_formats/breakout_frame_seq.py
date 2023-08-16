import os
from collections import deque
from pathlib import Path

import numpy as np
import cv2
from matplotlib import pyplot

from common import uint8_to_float32, resize
from deep_q.breakout_wrapper import env_name, num_actions

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from tensorflow import keras as ks

model_format_name = os.path.basename(__file__.removesuffix('.py'))
format_name = '_'.join([env_name, model_format_name])
model_path = os.path.join(Path.home(), 'My Drive', 'Models', format_name)
target_model_path = os.path.join(Path.home(), 'My Drive', 'Models', format_name + '_target')
data_path = os.path.join(Path.home(), 'Documents', 'Data', format_name)

frame_shape = (105, 80, 3)
frame_seq_len = 4
input_shape = (frame_seq_len,) + frame_shape


def create_q_model():
    inputs = ks.layers.Input(shape=input_shape)
    rescaled = ks.layers.Rescaling(1. / 255)(inputs)

    conv1_filters = 8

    conv_input = ks.layers.Input(shape=input_shape[1:])

    layer1 = ks.layers.Conv2D(conv1_filters, 8, strides=2, activation='relu')

    layer2 = ks.layers.Conv2D(16, 4, strides=2, activation='relu')

    layer3 = ks.layers.Conv2D(32, 3, strides=2, activation='relu')

    layer4 = ks.layers.Flatten()

    layer5 = ks.layers.Dense(512, activation='relu')

    convs = ks.Sequential([conv_input, layer1, layer2, layer3, layer4, layer5])

    time_distrib = ks.layers.TimeDistributed(convs)(rescaled)

    td_flattened = ks.layers.Flatten()(time_distrib)

    q_values = ks.layers.Dense(num_actions, activation=ks.activations.linear, kernel_initializer=ks.initializers.zeros)(td_flattened)

    return ks.Model(inputs=inputs, outputs=q_values)


def visualise(q, save=False):
    wasd_im = np.zeros((360, 640), dtype=np.uint8)
    frame_pairs = []

    font = cv2.FONT_HERSHEY_SIMPLEX

    fig, ax = pyplot.subplots(figsize=(7, 4))
    [nothing_plot] = ax.plot(0, 0, label='do nothing')
    [start_plot] = ax.plot(0, 0, label='press start')
    [right_plot] = ax.plot(0, 0, label='go right')
    [left_plot] = ax.plot(0, 0, label='go left')
    [actual_plot] = ax.plot(0, 0, label='actual')
    ax.set_title('Discounted Reward Prediction')
    ax.set_xlabel('timestep')
    ax.set_ylabel('discounted reward sum')
    ax.set_ylim((-1, 2))
    ax.legend()

    episode_frames = []

    while True:
        cv2.waitKey(1)

        q_item = q.get(block=True)

        if q_item is None:
            cv2.destroyAllWindows()
            return
        elif len(q_item) == 3:
            q_predictions, dr_sums, save_episode = q_item
            x_q = np.arange(len(q_predictions))
            x_d = np.arange(len(dr_sums))

            ax.set_xlim((0, len(dr_sums) - 1))

            nothing_plot.set_data(x_q, q_predictions[:, 0])
            start_plot.set_data(x_q, q_predictions[:, 1])
            right_plot.set_data(x_q, q_predictions[:, 2])
            left_plot.set_data(x_q, q_predictions[:, 3])
            actual_plot.set_data(x_d, dr_sums)

            fig.canvas.draw()
            (plot_width, plot_height) = fig.canvas.get_width_height()
            plot = np.frombuffer(fig.canvas.tostring_rgb(), np.uint8).reshape((plot_height, plot_width, 3))
            plot = cv2.cvtColor(plot, cv2.COLOR_BGR2RGB)  # reshape probably reverses RGB values

            cv2.imshow('Figure 1', plot)

            if save_episode:
                num_frames = len(episode_frames)
                print(f'{num_frames} frames')

                # Each frame of a gif is displayed for an integral number of hundredths of a second. Given a gif with a
                # non-standard framerate browsers will display it at a default 10fps. Couldn't get PIL to give me a gif
                # that had a high enough framerate.
                # imgs = [Image.fromarray(img, mode='RGB') for img in episode_frames]
                # imgs[0].save('agent_gameplay.webm', save_all=True, optimize=True, append_images=imgs[1:], loop=0)

                vid_w = episode_frames[0].shape[1]
                vid_h = episode_frames[0].shape[0]
                vp90cc = cv2.VideoWriter_fourcc('V', 'P', '9', '0')
                webm_writer = cv2.VideoWriter('agent_gameplay.webm', fourcc=vp90cc, fps=60, frameSize=(vid_w, vid_h))

                # The process for making a video that Twitter will accept sadly involves making a webm with cv2 and then
                # using an online file converter to convert it into an mp4. Browsers don't like mp4s created as below.
                # mp4vcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
                # webm_writer = cv2.VideoWriter('agent_gameplay.mp4', fourcc=mp4vcc, fps=60, frameSize=(vid_w, vid_h))

                print('Writing video')
                for frame in episode_frames:
                    webm_writer.write(frame)
                print('Video written')

                pyplot.savefig('q_prediction.png')

            episode_frames.clear()
        else:
            frame_array, action, random_action, q_values, reward = q_item

            for i in range(0, len(frame_array), 2):
                frame_pairs.append(np.hstack((frame_array[i],frame_array[i + 1])))

            frames = np.vstack(frame_pairs)
            frames = cv2.cvtColor(frames, cv2.COLOR_BGR2RGB)  # env.{reset,step} returns BGR for some reason :(
            frames = resize(frames, width=640)

            if save:
                frames_uint8 = frames.copy()  # .copy() because we want them to stay uint8s
                episode_frames.append(frames_uint8)

            frames = uint8_to_float32(frames)

            frames_convs = frames

            cv2.imshow('INPUT & CONV1', frames_convs)
            frame_pairs.clear()

            q_values = np.array(q_values)

            a0 = q_values[0]
            a1 = q_values[1]
            a2 = q_values[2]
            a3 = q_values[3]

            if random_action:
                streamer_text_colour = (255,)
                ai_text_colour = (85,)
            else:
                streamer_text_colour = (85,)
                ai_text_colour = (255,)

            cv2.putText(wasd_im, 'Random', (25, 50), font, 1, streamer_text_colour, 2)
            cv2.putText(wasd_im, 'AI: {}'.format(action), (25, 100), font, 1, ai_text_colour, 2)
            cv2.putText(wasd_im, 'Reward: {:.2f}'.format(reward), (25, 200), font, 1, (255,), 2)
            cv2.putText(wasd_im, '{: .2f} {: .2f} {: .2f} {: .2f}'.format(a0, a1, a2, a3), (25, 150), font, 1, (255,), 2)

            cv2.imshow('WASD', wasd_im)

            cv2.putText(wasd_im, 'Reward: {:.2f}'.format(reward), (25, 200), font, 1, (0,), 2)
            cv2.putText(wasd_im, 'AI: {}'.format(action), (25, 100), font, 1, (0,), 2)
            cv2.putText(wasd_im, '{: .2f} {: .2f} {: .2f} {: .2f}'.format(a0, a1, a2, a3), (25, 150), font, 1, (0,), 2)


class InputCollector:
    def __init__(self):
        self.frame_deque = deque(maxlen=frame_seq_len)

        for _ in range(frame_seq_len - 1):
            self.frame_deque.append(np.zeros(frame_shape, dtype=np.uint8))

    def get(self, frame):
        self.frame_deque.append(frame)
        return np.array(self.frame_deque)

    # def reset ...
