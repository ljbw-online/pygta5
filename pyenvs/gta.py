from collections import deque
from multiprocessing import Process, Queue
from socket import socket
from time import sleep, time

import cv2
import numpy as np
from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.specs import BoundedArraySpec
from tf_agents.trajectories.time_step import StepType, TimeStep

from common import resize

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


def render_loop(input_queue, output_queue):
    cap = cv2.VideoCapture(2, cv2.CAP_V4L2)  # Capture card
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        current_frame = read_or_reinitialise_capture(cap)

        if not input_queue.empty():
            item = input_queue.get()
            if item is None:
                input_queue.cancel_join_thread()
                output_queue.cancel_join_thread()
                cap.release()
                cv2.destroyAllWindows()
                return

            output_queue.put(current_frame)

        cv2.imshow('GTA', current_frame)
        if cv2.waitKey(16) == ord('q'):
            input_queue.put(None)


# This gets automatically turned into a TFPyEnvironment which in turn wraps the PyEnvironment in a BatchedPyEnvironment
# object. TFPyEnvironment overrides __getattr__ so that we can access attributes on the BatchedPyEnvironment object but
# BatchedPyEnvironment does not, so we can't access custom attributes on the PyEnvironment object.
class GTA(PyEnvironment):
    def __init__(self, discount=np.float32(0.99), render=True):
        super().__init__(handle_auto_reset=True)

        if render:
            self.rendering_in_q = Queue()
            self.rendering_out_q = Queue()
            self.rendering_process = Process(target=render_loop, args=(self.rendering_in_q, self.rendering_out_q))
            self.rendering_process.start()

        self.sock = socket()
        while True:
            try:
                self.sock.connect(("192.168.178.23", 7001))
                break
            except ConnectionRefusedError:
                print("ConnectionRefusedError")
                sleep(1)

        self._action_spec = BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=3, name='action')
        self._observation_spec = BoundedArraySpec(shape=(84, 84, 1), dtype=np.uint8, minimum=0, maximum=255,
                                                  name='observation')
        self._discount = discount
        self.current_human_frame = None

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def get_observation(self):
        self.rendering_in_q.put(True)
        return self.rendering_out_q.get(block=True)

    def _reset(self):
        self.sock.sendall(reinforcement_resumed_bytes)  # set control normals to 0
        self.sock.sendall(teleport_bytes)

        observation = self.read_or_reinitialise_capture()

        self.current_human_frame = observation
        observation = downscale(observation)

        return TimeStep(StepType.FIRST, reward=np.asarray(0, dtype=np.float32), discount=self._discount,
                        observation=observation)

    def send_action(self, action):
        # w nk s
        match action:
            case 0:
                self.sock.sendall(w_bytes)
            case 1:
                self.sock.sendall(nk_bytes)
            case 2:
                self.sock.sendall(s_bytes)
            case _:
                print('Invalid action: {}'.format(action))

    def _step(self, action):
        terminated = False

        self.send_action(action)
        recv_bytes = self.sock.recv(29)
        forward_speed = np.frombuffer(recv_bytes, dtype=np.float32, count=1, offset=1)
        forward_speed = forward_speed[0]

        observation = self.read_or_reinitialise_capture()
        self.current_human_frame = observation
        observation = downscale(observation)

        if terminated:
            step_type = StepType.LAST
        else:
            step_type = StepType.MID

        reward = abs(forward_speed)

        return TimeStep(step_type, reward=np.asarray(reward, dtype=np.float32), discount=self._discount,
                        observation=observation)

    def render(self, mode='rgb_array'):
        if self.current_human_frame is None:
            self.current_human_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        if mode == 'human':
            cv2.imshow('Breakout', self.current_human_frame)
            # waitKey in main loop
        else:
            return self.current_human_frame

    def close(self):
        cv2.destroyAllWindows()

        self.sock.sendall(reinforcement_paused_bytes)
        self.sock.close()

        self.rendering_in_q.put(None)
        self.rendering_process.join()


def test_env():
    loop_times = deque(maxlen=100)
    step_count = 0
    env = GTA()
    timestep = env.reset()
    while True:
        t = time()
        step_count += 1
        if timestep.step_type == StepType.LAST:
            timestep = env.reset()

        rendered_frame = env.render(mode='rgb_array')

        key = 0
        cv2.imshow('Observation', timestep.observation)
        cv2.imshow('Rendered', rendered_frame)
        key = cv2.waitKey(1)
        # print(key)

        # if env.current_human_frame is None:
        #     env.close()
        #     break

        action = 1  # no-keys

        if key == ord('q'):
            cv2.destroyAllWindows()
            env.close()
            break
        elif key == ord('i'):
            action = 0
        elif key == ord('k'):
            action = 2

        loop_times.append(time() - t)
        timestep = env.step(action)

        if step_count % 100 == 0:
            print(np.average(loop_times))


if __name__ == '__main__':
    test_env()
