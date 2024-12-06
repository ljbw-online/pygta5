from collections import deque
from multiprocessing import Process, Queue
from socket import socket
from time import sleep

import cv2
import keras
import numpy as np
from keras import layers

# Fails when this script is run directly. Apparently directly running a script which is in a package is 
# an antipattern. Use python -m environments.gta or run evaluate_environment.py.
from common import resize

env_name = 'GTA'

gamma = 0.99
epsilon_max = 1.0
max_steps_per_episode = 1_000

supervised_resumed_bytes = bytes([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
model_paused_bytes = bytes([2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
reinforcement_resumed_bytes = bytes([4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
reinforcement_paused_bytes = bytes([5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
reinforcement_model_resumed_bytes = bytes([3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
position_bytes = bytes([15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
state_bytes = bytes([23, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

w_bytes = bytes([22]) + np.array([1, 0, 0, 0], dtype=np.float32).tobytes()
wa_bytes = bytes([22]) + np.array([1, 1, 0, 0], dtype=np.float32).tobytes()
wd_bytes = bytes([22]) + np.array([1, 0, 0, 1], dtype=np.float32).tobytes()
s_bytes = bytes([22]) + np.array([0, 0, 1, 0], dtype=np.float32).tobytes()
sa_bytes = bytes([22]) + np.array([0, 1, 1, 0], dtype=np.float32).tobytes()
sd_bytes = bytes([22]) + np.array([0, 0, 1, 1], dtype=np.float32).tobytes()
a_bytes = bytes([22]) + np.array([0, 1, 0, 0], dtype=np.float32).tobytes()
d_bytes = bytes([22]) + np.array([0, 0, 0, 1], dtype=np.float32).tobytes()
nk_bytes = bytes([22]) + np.array([0, 0, 0, 0], dtype=np.float32).tobytes()


starting_position = np.array([1099, -265, 69], dtype=np.float32)
teleport_bytes = bytes([19, 0, 0, 0]) + starting_position.tobytes() + bytes([0])
# teleport_bytes = [0] * 17
# teleport_bytes[0] = 19
# teleport_bytes = bytes(teleport_bytes)

headings = [152, 332]

window_name = 'OpenCV'
action_labels = ['accelerate', 'nothing', 'brake/reverse', 'left', 'right']


def get_heading_bytes(heading):
    return bytes([21, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) + np.float32(heading).tobytes() + bytes([0])


def downscale(frame):
    frame = resize(frame, height=84, width=84)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = np.expand_dims(frame, axis=-1)
    return frame


def read_or_reinitialise_capture(cap):
    return_val, observation = cap.read()

    while return_val is False:
        print('Reinitialising capture')
        sleep(1)
        cap.release()
        cap = cv2.VideoCapture(2, cv2.CAP_V4L2)  # Capture card
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        return_val, observation = cap.read()

    return observation


def display_captures():
    for i in range(10):
        capture = get_new_capture(index=i)

        window_name = f'Capture {i}'
        return_val = True
        
        while return_val:
            return_val, frame = capture.read()

            cv2.imshow(window_name, frame)
            key_ord = cv2.waitKey(33)

            if key_ord == ord('q'):
                cv2.destroyWindow(window_name)
                break

    cv2.destroyAllWindows()


def get_new_capture(index=2):
    video_capture = cv2.VideoCapture(index, cv2.CAP_V4L2)
    video_capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return video_capture


def render_loop(in_q, out_q, testing):
    # from audio import CallbackPair
    video_capture = get_new_capture()

    # callback_pair = CallbackPair()
    # callback_pair.start()

    if testing:
        waitkey_duration = 1
    else:
        waitkey_duration = 12

    while in_q.empty():
        return_val, frame = video_capture.read()

        while return_val is False:
            print('Reinitialising capture')
            video_capture.release()
            sleep(1)
            video_capture = get_new_capture()
            return_val, frame = video_capture.read()

        if out_q.empty():
            out_q.put(frame)

        cv2.imshow(window_name, frame)

        # It appears that we do need to call waitKey in this subprocess
        cv2.waitKey(waitkey_duration)

    video_capture.release()
    # callback_pair.attempt_to_stop()

    in_q.cancel_join_thread()
    out_q.cancel_join_thread()


class Env:
    def __init__(self, render=True, stack=False, testing=False):
        self.name = env_name
        self.num_actions = 5
        self.max_return = 10_000
        self.random_eval_action = 1
        self.max_steps_per_episode = 1000
        self.evaluation_epsilon = 0.1

        self.in_q = Queue()
        self.frame_q = Queue()

        # Start displaying capture card so I can login to Windows if necessary
        self.rendering_process = Process(target=render_loop, args=(self.in_q, self.frame_q, testing),
                                         daemon=True)
        self.rendering_process.start()

        self.sock = socket()

        print('Connecting socket')
        self.connect_socket()
        print('Socket connected')

        self.timestep_dtype = np.dtype(
            [('observation', np.uint8, (84, 84)), ('action', np.int32), ('reward', np.float32)])

        # self.video_capture = None
        # self.initialise_capture()

        self.prev_position = None
        self.heading_number = 0
        self.first_step = True

        self.stack = stack

        if stack:
            self.frame_deque = deque(maxlen=4)

    def connect_socket(self):
        while True:
            try:
                self.sock.connect(("DESKTOP-I426F5T", 7001))
                break
            except Exception as ex:
                print(ex)
                sleep(1)

    def reset(self):
        while True:
            try:
                self.sock.sendall(reinforcement_resumed_bytes)  # sets control normals to 0
                self.sock.sendall(teleport_bytes)
                self.sock.sendall(get_heading_bytes(headings[self.heading_number]))
                break
            except Exception as ex:
                print(ex)
                self.sock.close()
                sleep(1)
                self.sock = socket()
                self.connect_socket()

        self.heading_number = (self.heading_number + 1) % len(headings)
        # self.prev_position = starting_position
        self.first_step = True

        frame = self.frame_q.get(block=True)
        # self.current_human_frame = frame

        # frame = downscale(frame)

        # if self.render:
        #     cv2.imshow(window_name, self.current_human_frame)
        #     cv2.waitKey(1)

        if self.stack:
            for _ in range(3):
                self.frame_deque.append(np.zeros((84, 84, 1), dtype=np.uint8))

            self.frame_deque.append(frame)
            return np.concatenate(self.frame_deque, axis=2)
        else:
            return frame

    def step(self, action):
        terminated = False

        while True:
            try:
                self.send_action(action)
                recv_bytes = self.sock.recv(29)
                break
            except Exception as ex:
                print(ex)
                self.sock.close()
                sleep(1)
                self.sock = socket()
                self.connect_socket()

        position = np.frombuffer(recv_bytes, dtype=np.float32, count=3, offset=5)
        submerged = bool(recv_bytes[17])

        if self.first_step:
            self.prev_position = position.copy()
            self.first_step = False

        if submerged:
            reward = np.float32(0)
            terminated = True
        else:
            reward = np.sqrt(np.sum((position - self.prev_position) ** 2)) / 100

        self.prev_position = position

        frame = self.frame_q.get(block=True)
        # self.current_human_frame = frame
        # frame = downscale(frame)

        if self.stack:
            self.frame_deque.append(frame)
            observation = np.concatenate(self.frame_deque, axis=2)
        else:
            observation = frame

        # if self.render:
        #     cv2.imshow('OpenCV', self.current_human_frame)
        #     cv2.waitKey(1)

        return observation, reward, terminated

    def send_action(self, action):
        # w nk s
        match action:
            case 0:
                self.sock.sendall(w_bytes)
            case 1:
                self.sock.sendall(nk_bytes)
            case 2:
                self.sock.sendall(s_bytes)
            case 3:
                self.sock.sendall(wa_bytes)
            case 4:
                self.sock.sendall(wd_bytes)
            case _:
                print('Invalid action: {}'.format(action))

    def render(self):
        # if self.current_human_frame is None:
        #     self.current_human_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        # cv2.imshow(window_name, self.current_human_frame)
        # return cv2.waitKey(1)
        return 0

    def create_q_net(self):
        inputs = layers.Input(shape=(84, 84, 4,))
        rescaling = layers.Rescaling(1. / 255)(inputs)

        conv1 = layers.Conv2D(32, 8, strides=4, activation='relu')(rescaling)
        conv2 = layers.Conv2D(64, 4, strides=2, activation='relu')(conv1)
        conv3 = layers.Conv2D(64, 3, strides=1, activation='relu')(conv2)

        flatten = layers.Flatten()(conv3)

        dense = layers.Dense(512, activation='relu')(flatten)
        q_values = layers.Dense(self.num_actions, activation='linear')(dense)

        return keras.Model(inputs=inputs, outputs=q_values)

    def pause(self):
        self.sock.sendall(reinforcement_resumed_bytes)

    def close(self):
        cv2.destroyAllWindows()

        self.sock.sendall(reinforcement_paused_bytes)
        self.sock.close()

        # self.video_capture.release()

        self.in_q.put(None)
        # Blocks because of SDL
        # self.rendering_process.join()


def test_env():
    episode_reward = 0
    key_ord = 0
    testing = False
    env = Env(testing=testing)
    terminated = False
    while True:
        if key_ord == ord('q'):
            break

        episode_reward = 0
        observation = env.reset()
        for step_count in range(max_steps_per_episode):
            if terminated:
                print('terminated')
                terminated = False
                break

            # observation = np.hstack(np.split(observation, 4, axis=2))
            cv2.imshow('Observation', observation)

            action = 1  # no-keys

            if testing:
                key_ord = cv2.waitKey(12)
            else:
                key_ord = cv2.waitKey(1)

            if key_ord == ord('q'):
                cv2.destroyAllWindows()
                env.close()
                break
            elif key_ord == ord('i'):
                action = 0
            elif key_ord == ord('j'):
                action = 3
            elif key_ord == ord('l'):
                action = 4
            elif key_ord == ord('k'):
                action = 2

            next_observation, reward, terminated = env.step(action)
            observation = next_observation

            episode_reward += reward

            print('{:.4f}'.format(reward))

        print(f'episode_reward: {episode_reward}')


if __name__ == '__main__':
    test_env()
