from time import time, sleep
import os
from pathlib import Path
from math import tanh, floor, ceil
from itertools import repeat, chain
from operator import add
from socket import socket

import numpy as np
import cv2
import tensorflow
import tensorflow.keras as ks

from common import (INPUT_WIDTH, INPUT_HEIGHT, get_gta_window, FORWARD, LEFT,
                    BRAKE, RIGHT, PressKey, W, S, A, D, ReleaseKey, K, imshow, I, KEYS, CORRECTION_KEYS, SCANCODES)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
np.set_printoptions(precision=3, floatmode='fixed', suppress=True)  # suppress stops scientific notation

tensorflow.compat.v1.enable_eager_execution()  # Necessary for .numpy() in TensorFlow 1

# DESCRIPTION NOT UPDATED RECENTLY
# Input: the game image, the key scores, a multihot_4 array representing the current key state

# Output: a multihot array of length four corresponding to whether W, A, S or D should be pressed or released. If the
# key is currently unpressed and the score goes above 0.5 then we press the key. If the key is pressed and the score
# goes below 0.5 then we release the key. The scores hence have different meanings for when a key is pressed and
# unpressed. A score being below 0.5 when the key is not pressed means "don't press the key" whereas when the key is
# pressed it means "release the key".

# The key scores: a four-element array recording the durations for which each of W, A, S and D have
# been pressed for in the recent past. Each element of this array, except the first, is incremented by one point per
# second whilst the key is pressed and decremented at the same rate whilst the key is not pressed, stopping at zero.
# The first element of the array, recording how long W has been pressed for, is reset to zero whenever S is released.
# This stops the first score from continually increasing.

# Balancing: data needs to be balanced across keys otherwise model will always press W regardless of image. It also
# needs to be balanced across pressing and releasing keys. For example for every frame on which I am pressing S and
# release S there needs to be a frame on which I am pressing S and *continue* to press S. Otherwise the model will
# constantly press and release keys.

MODEL_PATH = os.path.join(Path.home(), 'My Drive\\Models',  __name__)

# DATUM_SHAPE = (INPUT_HEIGHT + 1, INPUT_WIDTH, 3)
IMAGE_SHAPE = (INPUT_HEIGHT, INPUT_WIDTH, 3)

# W, A, S, D; multi-hot
OUTPUT_LENGTH = 4
OUTPUT_SHAPE = (OUTPUT_LENGTH,)

# DATA_TARGET = 0

# Number of distinct signals which the model can output.
# Frames on which I: didn't press any key, pressed A, pressed S, and pressed D, respectively.
# NUM_SIGNALS = 4
# MAX_DATA_PER_SIGNAL = 0  # Defining at module level to appease PyCharm, which also doesn't like it being None.

DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 360


# class SignalCounts:
#     forward_and_straight_on = 0
#     left = 0
#     brake = 0
#     right = 0
#
#     @classmethod
#     def list(cls):
#         return [cls.forward_and_straight_on, cls.left, cls.brake, cls.right]


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

    def set_current_key_scores_and_label(self, keys):
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
        self.set_current_key_scores_and_label(keys)

        if self.current_key_state != self.current_label:
            print(self.current_key_state, '!=', self.current_label)
            exit()
        # self.set_current_label(keys)


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

        if (self.index % round(self.array_length/100) == 0) and (self.index != 0):
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
        if self.correction_data:
            images_save_fn = correction_images_fn
            key_scores_save_fn = correction_key_scores_fn
            prev_key_states_save_fn = correction_prev_key_states_fn
            labels_save_fn = correction_labels_fn

            if os.path.isfile(correction_images_fn):
                print('Correction data already exists. Appending timestamp to filenames.')
                append_string = '_' + str(time())
                images_save_fn += append_string
                key_scores_save_fn += append_string
                prev_key_states_save_fn += append_string
                labels_save_fn += append_string

        else:
            images_save_fn = images_fn
            key_scores_save_fn = key_scores_fn
            prev_key_states_save_fn = prev_key_states_fn
            labels_save_fn = labels_fn

        print('Saving {} frames'.format(self.index))

        self.images = self.images[:self.index]
        np.save(images_save_fn, self.images)

        self.key_scores = self.key_scores[:self.index]
        np.save(key_scores_save_fn, self.key_scores)

        self.prev_key_states = self.prev_key_states[:self.index]
        np.save(prev_key_states_save_fn, self.prev_key_states)

        self.labels = self.labels[:self.index]
        np.save(labels_save_fn, self.labels.astype('float32'))

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


def correction_to_keypresses(keys):
    if FORWARD in keys and 'W' not in keys:
        PressKey(W)
    elif FORWARD not in keys and 'W' in keys:
        ReleaseKey(W)

    if LEFT in keys and 'A' not in keys:
        PressKey(A)
    elif LEFT not in keys and 'A' in keys:
        ReleaseKey(A)

    if BRAKE in keys and 'S' not in keys:
        PressKey(S)
    elif BRAKE not in keys and 'S' in keys:
        ReleaseKey(S)

    if RIGHT in keys and 'D' not in keys:
        PressKey(D)
    elif RIGHT not in keys and 'D' in keys:
        ReleaseKey(D)


images_fn = 'images_uint8.npy'
key_scores_fn = 'key_scores_float32.npy'
prev_key_states_fn = 'prev_key_states_float32.npy'
labels_fn = 'labels_float32.npy'

correction_images_fn = 'correction_images_uint8.npy'
correction_key_scores_fn = 'correction_key_scores_float32.npy'
correction_prev_key_states_fn = 'correction_prev_key_states_float32.npy'
correction_labels_fn = 'correction_labels_float32.npy'


def check_data(return_stuff=False, show=True, correction_data=False):
    if correction_data:
        images = np.load(correction_images_fn)
        key_scores = np.load(correction_key_scores_fn)
        prev_key_states = np.load(correction_prev_key_states_fn)
        labels = np.load(correction_labels_fn)
    else:
        images = np.load(images_fn)
        key_scores = np.load(key_scores_fn)
        prev_key_states = np.load(prev_key_states_fn)
        labels = np.load(labels_fn)

    print('images.shape', images.shape)
    print('key_scores.shape', key_scores.shape)
    print('prev_key_states.shape', prev_key_states.shape)
    print('labels.shape', labels.shape)

    forwards_and_straight_on = 0

    forwards_pressed = 0
    forwards_remaining_unpressed = 0
    forwards_released = 0
    forwards_remaining_pressed = 0

    lefts_pressed = 0
    lefts_remaining_unpressed = 0
    lefts_released = 0
    lefts_remaining_pressed = 0

    brakes_pressed = 0
    brakes_remaining_unpressed = 0
    brakes_released = 0
    brakes_remaining_pressed = 0

    rights_pressed = 0
    rights_remaining_unpressed = 0
    rights_released = 0
    rights_remaining_pressed = 0

    prev_key_states = prev_key_states.astype('bool')
    labels = labels.astype('bool')

    for i in range(len(labels)):
        p_ks = prev_key_states[i]
        lbl = labels[i]

        if (list(lbl) == [True, False, False, False])\
                and (list(p_ks) == [True, False, False, False]):
            forwards_and_straight_on += 1

        if not p_ks[0]:
            if lbl[0]:
                forwards_pressed += 1
            else:
                forwards_remaining_unpressed += 1
        elif p_ks[0]:
            if not lbl[0]:
                forwards_released += 1
            else:
                forwards_remaining_pressed += 1

        if not p_ks[1]:
            if lbl[1]:
                lefts_pressed += 1
            else:
                lefts_remaining_unpressed += 1
        elif p_ks[1]:
            if not lbl[1]:
                lefts_released += 1
            else:
                lefts_remaining_pressed += 1

        if not p_ks[2]:
            if lbl[2]:
                brakes_pressed += 1
            else:
                brakes_remaining_unpressed += 1
        elif p_ks[2]:
            if not lbl[2]:
                brakes_released += 1
            else:
                brakes_remaining_pressed += 1

        if not p_ks[3]:
            if lbl[3]:
                rights_pressed += 1
            else:
                rights_remaining_unpressed += 1
        elif p_ks[3]:
            if not lbl[3]:
                rights_released += 1
            else:
                rights_remaining_pressed += 1

    if return_stuff:
        return (images, key_scores, prev_key_states, labels,
                [forwards_pressed, forwards_remaining_unpressed, forwards_released, forwards_remaining_pressed,
                 lefts_pressed,    lefts_remaining_unpressed,    lefts_released,    lefts_remaining_pressed,
                 brakes_pressed,   brakes_remaining_unpressed,   brakes_released,   brakes_remaining_pressed,
                 rights_pressed,   rights_remaining_unpressed,   rights_released,   rights_remaining_pressed])
    else:
        print('forwards_and_straight_on', forwards_and_straight_on)

        print('forwards_pressed:', forwards_pressed, 'forwards_remaining_unpressed:', forwards_remaining_unpressed,
              'forwards_released:', forwards_released, 'forwards_remaining_pressed:', forwards_remaining_pressed)

        print('lefts_pressed:', lefts_pressed, 'lefts_remaining_unpressed:', lefts_remaining_unpressed,
              'lefts_released:', lefts_released, 'lefts_remaining_pressed:', lefts_remaining_pressed)

        print('brakes_pressed:', brakes_pressed, 'brakes_remaining_unpressed:', brakes_remaining_unpressed,
              'brakes_released:', brakes_released, 'brakes_remaining_pressed:', brakes_remaining_pressed)

        print('rights_pressed:', rights_pressed, 'rights_remaining_unpressed:', rights_remaining_unpressed,
              'rights_released:', rights_released, 'rights_remaining_pressed:', rights_remaining_pressed)

        if show:
            imshow(images, labels={'key score': key_scores, 'key state': prev_key_states, 'label': labels})
        else:
            pass


training_images_fn = 'training_images_float32.npy'
training_scores_fn = 'training_scores.npy'
training_prev_states_fn = 'training_prev_states.npy'
training_labels_fn = 'training_labels.npy'


def prepare_data(duplication_factor=4, noise_amplitude=8, show=False):
    # shuffling done with tensorflow

    images, key_scores, prev_key_states_bools, labels_bools, stats_list = check_data(return_stuff=True)

    if os.path.isfile(correction_images_fn):
        choice = input('Incorporate correction data? (y/N)')
    else:
        choice = 'n'

    if choice == 'y':
        # print('Tell check_data to load correction data this time')
        (corr_images, corr_key_scores, corr_prev_key_states_bools, corr_labels_bools,
         corr_stats_list) = check_data(return_stuff=True, correction_data=True)

        images = np.concatenate((images, corr_images))
        key_scores = np.concatenate((key_scores, corr_key_scores))
        prev_key_states_bools = np.concatenate((prev_key_states_bools, corr_prev_key_states_bools))
        labels_bools = np.concatenate((labels_bools, corr_labels_bools))

        stats_list = list(map(add, stats_list, corr_stats_list))

    w_p = []
    w_ru = []
    w_r = []
    w_rp = []

    a_p = []
    a_ru = []
    a_r = []
    a_rp = []

    s_p = []
    s_ru = []
    s_r = []
    s_rp = []

    d_p = []
    d_ru = []
    d_r = []
    d_rp = []

    data = zip(images, key_scores, prev_key_states_bools, labels_bools)

    for datum in data:
        p_ks = datum[2]
        lbl = datum[3]

        if not p_ks[0]:
            if lbl[0]:
                w_p.append(datum)
            else:
                w_ru.append(datum)
        elif p_ks[0]:
            if not lbl[0]:
                w_r.append(datum)
            else:
                w_rp.append(datum)

        if not p_ks[1]:
            if lbl[1]:
                a_p.append(datum)
            else:
                a_ru.append(datum)
        elif p_ks[1]:
            if not lbl[1]:
                a_r.append(datum)
            else:
                a_rp.append(datum)

        if not p_ks[2]:
            if lbl[2]:
                s_p.append(datum)
            else:
                s_ru.append(datum)
        elif p_ks[2]:
            if not lbl[2]:
                s_r.append(datum)
            else:
                s_rp.append(datum)

        if not p_ks[3]:
            if lbl[3]:
                d_p.append(datum)
            else:
                d_ru.append(datum)
        elif p_ks[3]:
            if not lbl[3]:
                d_r.append(datum)
            else:
                d_rp.append(datum)

    print(len(w_p))
    print(len(w_ru))
    print(len(w_r))
    print(len(w_rp))

    print(len(a_p))
    print(len(a_ru))
    print(len(a_r))
    print(len(a_rp))

    print(len(s_p))
    print(len(s_ru))
    print(len(s_r))
    print(len(s_rp))

    print(len(d_p))
    print(len(d_ru))
    print(len(d_r))
    print(len(d_rp))

    training_data = []
    rng = np.random.default_rng().integers

    min_stat = min(stats_list)
    min_datum_list_len = min_stat * duplication_factor

    for datum_list in [w_p, w_ru, w_r, w_rp, a_p, a_ru, a_r, a_rp, s_p, s_ru, s_r, s_rp, d_p, d_ru, d_r, d_rp]:
        print()
        print('original length', len(datum_list))
        needed_duplication = len(datum_list) < min_datum_list_len
        datum_list = sum(repeat(datum_list, ceil(min_datum_list_len / len(datum_list))), [])
        print('after duplication', len(datum_list))
        jump = floor(len(datum_list) / min_datum_list_len)
        datum_list = datum_list[:(min_datum_list_len * jump)]
        print('after truncation', len(datum_list))

        if needed_duplication:
            for i, datum in enumerate(datum_list):
                amp = round(noise_amplitude / 2)
                im = datum[0].astype('int16')
                im = (im + rng(- amp, amp, im.size, dtype='int16').reshape(im.shape)).astype('uint8')
                datum_list[i] = (im,) + datum[1:]

        for i in range(0, len(datum_list), jump):
            training_data.append(datum_list[i])

    # images = np.zeros((len(training_data),) + images.shape[1:], dtype='float32')
    images = np.memmap(training_images_fn, dtype='float32', mode='w+', shape=(len(training_data),) + images.shape[1:])
    key_scores = np.zeros((len(training_data),) + key_scores.shape[1:], dtype='float32')
    prev_key_states = np.zeros((len(training_data),) + prev_key_states_bools.shape[1:], dtype='float32')
    labels = np.zeros((len(training_data),) + labels_bools.shape[1:], dtype='float32')

    print(images.shape)
    print(key_scores.shape)
    print(prev_key_states.shape)
    print(labels.shape)

    for i in range(len(training_data)):
        image = training_data[i][0]
        key_score = training_data[i][1]
        prev_key_state = training_data[i][2]
        label = training_data[i][3]

        images[i] = image.astype('float32') / 255  # convert to float32 and rescale
        key_scores[i] = key_score
        prev_key_states[i] = prev_key_state.astype('float32')  # convert back from bools
        labels[i] = label.astype('float32')

    if show:
        imshow((images * 255).astype('uint8'), labels={'key score': key_scores, 'key state': prev_key_states,
                                                       'label': labels, 'index': range(len(labels))})

    choice = input('Save? (y/N)')
    if choice == 'y':
        # np.save(training_images_fn, images)
        images.flush()
        np.save(training_scores_fn, key_scores)
        np.save(training_prev_states_fn, prev_key_states)
        np.save(training_labels_fn, labels)


def merge_correction_data():
    # ASSUMES ONLY ONE EXTRA SET OF ARRAYS
    for fn in os.listdir():
        if fn.startswith('correction') and fn.endswith('.npy'):
            array_contents = fn[11:17]
            if fn[-10:-4].isnumeric():
                if array_contents == 'images':
                    extra_images = np.load(fn)
                elif array_contents == 'key_sc':
                    extra_key_scores = np.load(fn)
                elif array_contents == 'prev_k':
                    extra_prev_key_states = np.load(fn)
                elif array_contents == 'labels':
                    extra_labels = np.load(fn)
                else:
                    print(fn)
            else:
                if array_contents == 'images':
                    images = np.load(fn)
                elif array_contents == 'key_sc':
                    key_scores = np.load(fn)
                elif array_contents == 'prev_k':
                    prev_key_states = np.load(fn)
                elif array_contents == 'labels':
                    labels = np.load(fn)
                else:
                    print(fn)

    # print(extra_images.shape, extra_key_scores.shape, extra_prev_key_states.shape, extra_labels.shape)

    images = np.concatenate((images, extra_images))
    key_scores = np.concatenate((key_scores, extra_key_scores))
    prev_key_states = np.concatenate((prev_key_states, extra_prev_key_states))
    labels = np.concatenate((labels, extra_labels))

    print(images.shape, key_scores.shape, prev_key_states.shape, labels.shape)

    choice = input('Save? y/N')
    if choice == 'y':
        np.save(correction_images_fn, images)
        np.save(correction_key_scores_fn, key_scores)
        np.save(correction_prev_key_states_fn, prev_key_states)
        np.save(correction_labels_fn, labels)


def train_new_model(epochs=1):
    # Images converted to float32 and rescaled at saving time

    image_input = ks.Input(shape=(INPUT_HEIGHT, INPUT_WIDTH, 3), name='image_input')
    x = ks.layers.Conv2D(6, 4, padding='same', activation='relu', data_format='channels_last',
                         name='1st_conv')(image_input)
    x = ks.layers.MaxPooling2D(data_format='channels_last')(x)
    x = ks.layers.Conv2D(12, 4, padding='same', activation='relu', data_format='channels_last',
                         name='2nd_conv')(x)
    x = ks.layers.MaxPooling2D(data_format='channels_last')(x)
    x = ks.layers.Conv2D(24, 4, padding='same', activation='relu', data_format='channels_last',
                         name='3rd_conv')(x)
    x = ks.layers.MaxPooling2D(data_format='channels_last')(x)
    x = ks.layers.Flatten()(x)
    x = ks.layers.Dense(16, activation='relu', name='penult_dense')(x)

    scores_input = ks.Input(shape=(4,), name='scores_input')
    y = ks.layers.Dense(8, activation='relu')(scores_input)

    states_input = ks.Input(shape=(4,), name='states_input')
    w = ks.layers.Dense(8, activation='relu')(states_input)

    z = ks.layers.concatenate([x, y, w])
    model_output = ks.layers.Dense(OUTPUT_LENGTH, activation='sigmoid', name='model_output')(z)

    model = ks.Model(inputs=[image_input, scores_input, states_input], outputs=model_output)

    print(model.summary())

    # images = np.load(training_images_fn, mmap_mode='r')
    # np.memmap would not load this 16GB array without being passed the exact shape of the array.
    images = np.memmap(training_images_fn, dtype='float32', mode='r', shape=(99392, 90, 160, 3))
    scores = np.load(training_scores_fn)
    states = np.load(training_prev_states_fn)
    labels = np.load(training_labels_fn)

    model.compile(loss=ks.losses.MeanSquaredError(), metrics=['accuracy'])

    model.fit({'image_input': images, 'scores_input': scores, 'states_input': states},
              {'model_output': labels},
              epochs=epochs,
              shuffle=True,
              validation_split=0.2)

    while True:
        choice = input('Enter a number N to continue training for N epochs\n'
                       'Enter s to save and quit\n'
                       'Enter c to save the model and train for another epoch\n'
                       'Enter q to quit without saving\n'
                       '(N/s/q)\n')

        if choice.isnumeric():
            model.fit({'image_input': images, 'scores_input': scores, 'states_input': states},
                      {'model_output': labels},
                      epochs=int(choice),
                      shuffle=True,
                      validation_split=0.2)
        elif choice == 's':
            model.save(MODEL_PATH)
            return
        elif choice == 'c':
            model.save(MODEL_PATH)
            model.fit({'image_input': images, 'scores_input': scores, 'states_input': states},
                      {'model_output': labels},
                      epochs=1,
                      shuffle=True,
                      validation_split=0.2)
        elif choice == 'q':
            return


def prediction_to_key_presses(pred):
    threshold = 0.5

    if pred[0] > threshold:
        PressKey(W)
    else:
        ReleaseKey(W)

    if pred[1] > threshold:
        PressKey(A)
    else:
        ReleaseKey(A)

    if pred[2] > threshold:
        PressKey(S)
    else:
        ReleaseKey(S)

    if pred[3] > threshold:
        PressKey(D)
    else:
        ReleaseKey(D)


import warnings


def rescale(np_array):
    array_max = np.max((0.1, np.max(np_array)))  # stops zero denom below
    array_min = np.min(np_array)

    # warnings.simplefilter('error')
    # try:
    #     np_array = (np_array - array_min) / (array_max - array_min)
    # except RuntimeWarning:
    #     print('RuntimeWarning', np_array.shape)
    # warnings.simplefilter('default')

    np_array = (np_array - array_min) / (array_max - array_min)

    return np_array


font = cv2.FONT_HERSHEY_SIMPLEX


def display_features(q_param):
    while True:
        # start_time = time()

        # sleep(1./15)

        while True:
            current_key_state, current_key_durations, model_layers, correcting = q_param.get()
            if q_param.qsize() == 0:
                break
            else:
                pass

        if current_key_state is None:
            print('Returning subprocess')
            break
        else:
            pass

        conv1 = rescale(model_layers[0][0].numpy())
        conv1_f1 = conv1[:, :, 0]
        conv1_f2 = conv1[:, :, 1]
        conv1_f3 = conv1[:, :, 2]
        conv1_f4 = conv1[:, :, 3]
        conv1_f5 = conv1[:, :, 4]
        conv1_f6 = conv1[:, :, 5]

        conv2 = rescale(model_layers[1][0].numpy())
        conv2_f1 = conv2[:, :, 0]
        conv2_f2 = conv2[:, :, 1]
        conv2_f3 = conv2[:, :, 2]
        conv2_f4 = conv2[:, :, 3]
        conv2_f5 = conv2[:, :, 4]
        conv2_f6 = conv2[:, :, 5]

        conv1_f12 = np.concatenate((conv1_f1, conv1_f2), axis=1)
        conv1_f34 = np.concatenate((conv1_f3, conv1_f4), axis=1)
        conv1_f56 = np.concatenate((conv1_f5, conv1_f6), axis=1)
        conv_img1 = np.concatenate((conv1_f12, conv1_f34, conv1_f56))

        conv2_f12 = np.concatenate((conv2_f1, conv2_f2), axis=1)
        conv2_f34 = np.concatenate((conv2_f3, conv2_f4), axis=1)
        conv2_f56 = np.concatenate((conv2_f5, conv2_f6), axis=1)
        conv_img2 = np.concatenate((conv2_f12, conv2_f34, conv2_f56))

        conv_img2 = cv2.resize(conv_img2, (320, 270), interpolation=cv2.INTER_NEAREST)

        conv_features = np.concatenate((conv_img1, conv_img2))  # , conv_img3))
        conv_features = cv2.resize(conv_features, (640, 900), interpolation=cv2.INTER_NEAREST)

        if correcting:
            # 1.0 is white because conv_features is still float32
            cv2.putText(conv_features, 'Correcting', (128, 240), font, 2, (1.0,), 3)
            cv2.putText(conv_features, 'Correcting', (128, 600), font, 2, (1.0,), 3)
            cv2.putText(conv_features, 'Correcting', (128, 960), font, 2, (1.0,), 3)

        conv_features = (conv_features * 255).astype('uint8')

        penult_dense = rescale(model_layers[2][0].numpy())
        penult_dense = (penult_dense * 255).astype('uint8')
        penult_dense = np.expand_dims(penult_dense, 0)
        penult_dense = cv2.resize(penult_dense, (640, 30), interpolation=cv2.INTER_NEAREST)

        scores = (np.array(current_key_durations) * 255)
        for i in range(4):
            scores[i] = min(255., scores[i])
        scores = scores.astype('uint8')
        scores = np.expand_dims(scores, 0)
        scores = cv2.resize(scores, (640, 50), interpolation=cv2.INTER_NEAREST)

        if correcting:
            wasd = np.array(current_key_state).astype('uint8') * 255
            wasd_colours = list(map(lambda x: (int(x),), wasd))  # tuples containing value cast to python int

        else:
            wasd = (model_layers[-1][0].numpy() * 255).astype('uint8')
            wasd_colours = list(map(lambda x: (int(x),), wasd))  # tuples containing value cast to python int

        for i, colour in enumerate(wasd_colours):
            if colour[0] > 127:
                wasd_colours[i] = (64,)

        wasd = np.expand_dims(wasd, 0)
        wasd = cv2.resize(wasd, (640, 50), interpolation=cv2.INTER_NEAREST)
        text_top_left = (round(640 * 0.1), round(50 * 0.9))
        cv2.putText(wasd, 'W', text_top_left, font, 2, wasd_colours[0], 3)
        text_top_left = (round(640 * 0.35), round(50 * 0.9))
        cv2.putText(wasd, 'A', text_top_left, font, 2, wasd_colours[1], 3)
        text_top_left = (round(640 * 0.60), round(50 * 0.9))
        cv2.putText(wasd, 'S', text_top_left, font, 2, wasd_colours[2], 3)
        text_top_left = (round(640 * 0.85), round(50 * 0.9))
        cv2.putText(wasd, 'D', text_top_left, font, 2, wasd_colours[3], 3)

        penult_scores_wasd = np.concatenate((penult_dense, scores, wasd))
        # print(penult_scores_wasd.dtype, conv_features.dtype)
        features_scores_keys = np.concatenate((conv_features, penult_scores_wasd))

        cv2.imshow('FEATURES, SCORES and KEYS PRESSED', features_scores_keys)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            return

        # print(round(1 / (time() - start_time)))


layer_outputs = [0, 0, 0, 0]  # PyCharm appeasement


class ModelRunner(InputCollector):
    def __init__(self, q_param):
        super().__init__()
        # from tensorflow.keras.models import load_model, Model
        model = ks.models.load_model(MODEL_PATH)
        model = ks.models.Model(inputs=model.inputs, outputs=[model.get_layer('1st_conv').output,
                                                              model.get_layer('2nd_conv').output,
                                                              model.get_layer('penult_dense').output,
                                                              model.get_layer('model_output').output])
        self.model = model
        self.stuck = False
        self.stuck_time = 0
        self.dormant = False
        self.most_recent_not_coasting = time()

        self.q = q_param
        self.q_counter = 0

        self.sock = socket()

        while True:
            try:
                self.sock.connect(("127.0.0.1", 7001))
                break
            except ConnectionRefusedError:
                print("Connection Refused")
                sleep(1)

        # self.sock.sendall((0).to_bytes(1, byteorder='big', signed=False))

    def run_model(self, keys, correcting):
        global layer_outputs
        self.collect_input(keys)

        if not self.correction_data:
            for i in range(1, 4):
                if (self.current_key_durations[i]) > 3.0:
                    print('Resetting key score {}'.format(i))
                    self.current_key_durations[i] = 0
                    self.key_duration_mins[i] = 0
                    self.key_press_times[i] = self.t
                    self.key_duration_maxes[i] = 0

        if (not correcting) or self.correction_data:
            layer_outputs = self.model([np.expand_dims(self.current_frame.astype('float32') / 255, 0),
                                        np.expand_dims(np.array(self.current_key_scores, dtype='float32'), 0),
                                        np.expand_dims(np.array(self.previous_key_state, dtype='float32'), 0)])

            # prediction_and_correction_to_key_presses(layer_outputs[-1][0], keys, self.dormant)

            try:
                if time() - self.most_recent_not_coasting > 2.0:
                    self.sock.sendall(np.array([1, 0, 0, 0], dtype=np.float32).tobytes())
                else:
                    self.sock.sendall(layer_outputs[-1][0].numpy().tobytes())
            except ConnectionResetError:
                print("ConnectionResetError")
                while True:
                    self.sock = socket()
                    try:
                        self.sock.connect(("127.0.0.1", 7001))
                        break
                    except ConnectionRefusedError:
                        print("ConnectionRefusedError")
                        sleep(1)

                # print(layer_outputs[-1][0].numpy())

        # if not correcting:
        # elif self.correction_data:
        #     prediction_and_correction_to_key_presses(layer_outputs[-1][0], keys, self.dormant)

        # self.q_counter = (self.q_counter + 1) % 2
        # if self.q_counter == 0:
        #     self.q.put((self.current_key_state, self.current_key_durations, layer_outputs, correcting))

        if not self.correction_data:
            prev_output = layer_outputs[-1][0]
            if (prev_output[0] > 0.5) or (prev_output[2] > 0.5):
                self.dormant = False
                self.most_recent_not_coasting = time()
            else:
                self.dormant = True

            dormant_time = time() - self.most_recent_not_coasting
            if self.dormant:
                if (dormant_time > 3.0) and (dormant_time < 3.5) and ('I' not in keys):
                    print('dormant')
                    self.most_recent_not_coasting = time()
                    # PressKey(I)
                elif (dormant_time >= 1.0) and ('I' in keys):
                    pass
                    # ReleaseKey(I)

    def quit_model(self):
        self.q.put((None, None, None, None))
        # self.sock.sendall((1).to_bytes(1, byteorder='big', signed=False))  # Send zero to ljbw_bot
        self.sock.sendall(bytes([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
        self.sock.close()

    def pause_model(self, paused):
        self.sock.sendall(bytes([paused, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))


def prediction_and_correction_to_key_presses(pred, keys, dormant):
    threshold = 0.5
    # opp_keys = ['K', 'L', 'I', 'J']  # opposite keys
    # don't press left or right if I'm pressing forward

    forward_but_not_dormant = (FORWARD not in keys) and not dormant

    if 'W' in keys:
        if (FORWARD not in keys) and (pred[0] <= threshold):
            ReleaseKey(W)
    else:
        if (FORWARD in keys) or ((pred[0] > threshold) and (BRAKE not in keys)):
            PressKey(W)

    if 'A' in keys:
        if (LEFT not in keys) and (pred[1] <= threshold):
            ReleaseKey(A)
    else:
        if (LEFT in keys) or ((pred[1] > threshold) and (RIGHT not in keys) and forward_but_not_dormant):
            PressKey(A)

    if 'S' in keys:
        if (BRAKE not in keys) and (pred[2] <= threshold):
            ReleaseKey(S)
    else:
        if (BRAKE in keys) or ((pred[2] > threshold) and (FORWARD not in keys)):
            PressKey(S)

    if 'D' in keys:
        if (RIGHT not in keys) and (pred[3] <= threshold):
            ReleaseKey(D)
    else:
        if (RIGHT in keys) or ((pred[3] > threshold) and (LEFT not in keys) and forward_but_not_dormant):
            PressKey(D)

    # Doesn't work, possibly because keys are being pressed or released too rapidly.
    # for wasd_key, corr_key, model_pred, scancode, opp_key in zip(KEYS, CORRECTION_KEYS, pred, SCANCODES, opp_keys):
    #     if wasd_key in keys:
    #         if (corr_key not in keys) and (model_pred.numpy() <= threshold):
    #             ReleaseKey(scancode)
    #             print('releasing' + corr_key)
    #     else:
    #         if (corr_key in keys) or ((model_pred.numpy() > threshold) and (opp_key not in keys)):
    #             PressKey(scancode)
    #             print('pressing' + corr_key)
