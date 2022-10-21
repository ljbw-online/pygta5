from time import time
from math import tanh

import numpy as np
import cv2

from common import (get_gta_window, INPUT_WIDTH, INPUT_HEIGHT, DISPLAY_WIDTH, DISPLAY_HEIGHT, FORWARD, LEFT, BRAKE,
                    RIGHT, PressKey, W, A, S, D, ReleaseKey)

from . import DATA_DIRECTORY  # this script still currently saves in current dir

IMAGE_SHAPE = (INPUT_HEIGHT, INPUT_WIDTH, 3)
OUTPUT_LENGTH = 4
OUTPUT_SHAPE = (OUTPUT_LENGTH,)


def get_frame():
    frame = get_gta_window()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
    frame = cv2.resize(frame, (INPUT_WIDTH, INPUT_HEIGHT))
    return frame


class InputCollector:

    def __init__(self):
        self.current_frame = get_frame()
        self.current_key_scores = [0., 0., 0., 0.]
        self.current_key_state = [False, False, False, False]
        self.current_label = [False, False, False, False]

        self.previous_key_state = [False, False, False, False]

        self.t = time()
        self.current_key_durations = [0., 0., 0., 0.]
        self.key_press_times = [0., 0., 0., 0.]
        self.key_release_times = [0., 0., 0., 0.]
        self.key_duration_maxes = [0., 0., 0., 0.]
        self.key_duration_mins = [0., 0., 0., 0.]

        self.correction_data = False
        self.corrected_since_unpausing = False

    def set_current_and_prev_key_state(self, keys):
        self.previous_key_state = self.current_key_state.copy()

        self.current_key_state[0] = 'W' in keys
        self.current_key_state[1] = 'A' in keys
        self.current_key_state[2] = 'S' in keys
        self.current_key_state[3] = 'D' in keys

    def set_current_key_scores_and_label(self):
        self.t = time()

        for i in range(4):
            if self.current_key_state[i]:
                # If a key is down but was not down on the previous frame then it has been pressed.
                if not self.previous_key_state[i]:  # Pressed
                    self.current_key_state[i] = True
                    self.key_press_times[i] = self.t
                    self.key_duration_mins[i] = self.current_key_durations[i]

                    # Release W when we press S
                    # if i == 2:
                    #     self.current_key_state[0] = False
                    #     self.key_release_times[0] = self.t

                # Pressed or Remaining Pressed
                self.current_key_durations[i] = self.key_duration_mins[i] + (self.t - self.key_press_times[i])
                self.current_label[i] = True
            elif not self.current_key_state[i]:
                if self.previous_key_state[i]:  # Released
                    self.current_key_state[i] = False
                    self.key_release_times[i] = self.t
                    # Set the key_score_max when we release the key
                    self.key_duration_maxes[i] = self.current_key_durations[i]

                    # Releasing S has to look like a press of W so that the forward score starts again from zero
                    if i == 2:
                        self.key_duration_maxes[0] = 0
                        self.key_release_times[0] = self.t

                # Released or Remaing Unpressed
                self.current_key_durations[i] = max(
                    0., self.key_duration_maxes[i] - (self.t - self.key_release_times[i])
                )
                self.current_label[i] = False

            self.current_key_scores[i] = tanh(0.5 * self.current_key_durations[i])

    # def set_current_label(self, keys):
    #     self.current_label[0] = FORWARD in keys
    #     self.current_label[1] = LEFT in keys
    #     self.current_label[2] = BRAKE in keys
    #     self.current_label[3] = RIGHT in keys

    def collect_input(self, keys):
        self.current_frame = get_frame()
        self.set_current_and_prev_key_state(keys)
        self.set_current_key_scores_and_label()

        if self.current_key_state != self.current_label:
            print(self.current_key_state, '!=', self.current_label)
            exit()
        # self.set_current_label(keys)


images_fn = 'images_uint8.npy'
key_scores_fn = 'key_scores_float32.npy'
prev_key_states_fn = 'prev_key_states_float32.npy'
labels_fn = 'labels_float32.npy'


class DataCollector(InputCollector):

    def __init__(self, array_length):
        super().__init__()

        self.array_length = array_length

        self.images = np.zeros((array_length,) + IMAGE_SHAPE, dtype='uint8')
        self.key_scores = np.zeros((array_length, 4), dtype='float32')
        self.prev_key_states = np.zeros((array_length, 4), dtype='float32')
        self.labels = np.zeros((array_length,) + OUTPUT_SHAPE, dtype='float32')

        self.index = 0
        # W pressed, W remaining unpressed, W released, W remaining pressed, etc.
        self.signal_counts = [0., 0., 0., 0., 0., 0., 0., 0.,
                              0., 0., 0., 0., 0., 0., 0., 0.]

        self.fso = 0
        self.skip_counter = 0
        self.SKIP_COUNTER_MAX = 4

        self.correcting = False

    def conditionally_add_datum(self):
        decision = True

        if ((self.previous_key_state == [True, False, False, False])
                and (self.current_key_state == [True, False, False, False])):

            if self.skip_counter == 0:
                self.signal_counts[0 * 4 + 3] += 1  # w remaining pressed
                self.signal_counts[1 * 4 + 1] += 1  # a remaining unpressed
                self.signal_counts[2 * 4 + 1] += 1  # s remaining unpressed
                self.signal_counts[3 * 4 + 1] += 1  # d remaining unpressed
                self.fso += 1
            else:
                decision = False

            self.skip_counter = (self.skip_counter + 1) % self.SKIP_COUNTER_MAX

        else:
            for i in range(4):
                if not self.previous_key_state[i] and self.current_key_state[i]:  # Key Pressed
                    self.signal_counts[i * 4] += 1
                elif not self.previous_key_state[i] and not self.current_key_state[i]:  # Key Remaining Unpressed
                    self.signal_counts[i * 4 + 1] += 1
                elif self.previous_key_state[i] and not self.current_key_state[i]:  # Key Released
                    self.signal_counts[i * 4 + 2] += 1
                elif self.previous_key_state[i] and self.current_key_state[i]:  # Key Remaining Pressed
                    self.signal_counts[i * 4 + 3] += 1

        if decision:
            self.images[self.index] = self.current_frame
            self.key_scores[self.index] = self.current_key_scores
            self.prev_key_states[self.index] = self.previous_key_state
            self.labels[self.index] = self.current_label

            self.index += 1

            # print(self.current_label)

        if (self.index % round(self.array_length / 100) == 0) and (self.index != 0):
            self.print_signals()

    def collect_datum(self, keys):
        self.collect_input(keys)
        self.conditionally_add_datum()
        if not self.correction_data:
            self.display_frame()

    def display_frame(self):
        display_frame = cv2.resize(self.current_frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT),
                                   interpolation=cv2.INTER_NEAREST)
        text_top_left = (round(DISPLAY_WIDTH * 0.1), round(DISPLAY_HEIGHT * 0.9))
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(display_frame, 'Collecting data', text_top_left, font, 2, (255, 255, 255), 3)
        # cv2.imshow('ALANN', display_frame)

        scores = (np.array(self.current_key_scores) * 255)
        # for i in range(4):
        #     scores[i] = min(255., scores[i])
        scores = scores.astype('uint8')
        scores = np.expand_dims(scores, 0)
        scores = cv2.cvtColor(scores, cv2.COLOR_GRAY2RGB)
        scores = cv2.resize(scores, (640, 50), interpolation=cv2.INTER_NEAREST)

        wasd = np.array(self.current_key_state).astype('uint8')
        wasd = wasd * 255
        wasd = np.expand_dims(wasd, 0)
        wasd = cv2.cvtColor(wasd, cv2.COLOR_GRAY2RGB)
        wasd = cv2.resize(wasd, (640, 50), interpolation=cv2.INTER_NEAREST)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_top_left = (round(640 * 0.1), round(50 * 0.9))
        cv2.putText(wasd, 'W', text_top_left, font, 2, (128, 128, 128), 3)
        text_top_left = (round(640 * 0.35), round(50 * 0.9))
        cv2.putText(wasd, 'A', text_top_left, font, 2, (128, 128, 128), 3)
        text_top_left = (round(640 * 0.60), round(50 * 0.9))
        cv2.putText(wasd, 'S', text_top_left, font, 2, (128, 128, 128), 3)
        text_top_left = (round(640 * 0.85), round(50 * 0.9))
        cv2.putText(wasd, 'D', text_top_left, font, 2, (128, 128, 128), 3)

        display_frame = np.concatenate((display_frame, scores, wasd))
        cv2.imshow('KEY SCORES AND KEYS PRESSED', display_frame)

        # waitKey has to be called between imshow calls
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            return

    def save(self):
        print('Saving {} frames'.format(self.index))

        self.images = self.images[:self.index]
        np.save(images_fn, self.images)

        self.key_scores = self.key_scores[:self.index]
        np.save(key_scores_fn, self.key_scores)

        self.prev_key_states = self.prev_key_states[:self.index]
        np.save(prev_key_states_fn, self.prev_key_states)

        self.labels = self.labels[:self.index]
        np.save(labels_fn, self.labels.astype('float32'))

    def stop_session_decision(self):
        # Always increment index *after* deciding whether to save datum
        return self.index == self.array_length

    def print_signals(self):
        signal_ints = list(map(int, self.signal_counts))
        print()
        print('fso', self.fso)
        print('index', self.index)
        print(signal_ints[0:4])
        print(signal_ints[4:8])
        print(signal_ints[8:12])
        print(signal_ints[12:16])
