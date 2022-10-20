from time import time, sleep
import os
from math import tanh, floor, ceil
from itertools import repeat, product
from operator import add

import numpy as np
import cv2
import tensorflow
import tensorflow.keras as ks

from common import (INPUT_WIDTH, INPUT_HEIGHT, get_gta_window, FORWARD, LEFT,
                    BRAKE, RIGHT, PressKey, W, A, S, D, ReleaseKey, I)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
np.set_printoptions(precision=3, floatmode='fixed', suppress=True)  # suppress stops scientific notation

tensorflow.compat.v1.enable_eager_execution()  # Necessary for .numpy() in TensorFlow 1

MODEL_NAME = __name__

# W, A, S, D; multi-hot
OUTPUT_LENGTH = 4
OUTPUT_SHAPE = (OUTPUT_LENGTH,)

DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 360

FONT = cv2.FONT_HERSHEY_SIMPLEX


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

        self.previous_key_states = [False, False, False, False]

        self.t = time()
        self.current_key_durations = [0., 0., 0., 0.]
        self.key_press_times = [0., 0., 0., 0.]
        self.key_release_times = [0., 0., 0., 0.]
        self.key_duration_maxes = [0., 0., 0., 0.]
        self.key_duration_mins = [0., 0., 0., 0.]

        self.correction_data = False

    def set_current_and_prev_key_state(self, keys):
        self.previous_key_states = self.current_key_state.copy()

        self.current_key_state[0] = 'W' in keys
        self.current_key_state[1] = 'A' in keys
        self.current_key_state[2] = 'S' in keys
        self.current_key_state[3] = 'D' in keys

    def set_current_key_scores_and_label(self, keys):
        self.t = time()

        for i in range(4):
            if self.current_key_state[i]:
                # If a key is down but was not down on the previous frame then it has been pressed.
                if not self.previous_key_states[i]:  # Pressed
                    self.current_key_state[i] = True
                    self.key_press_times[i] = self.t
                    self.key_duration_mins[i] = self.current_key_durations[i]

                # Pressed or Remaining Pressed
                self.current_key_durations[i] = self.key_duration_mins[i] + (self.t - self.key_press_times[i])
                self.current_label[i] = True

            elif not self.current_key_state[i]:
                if self.previous_key_states[i]:  # Released
                    self.current_key_state[i] = False
                    self.key_release_times[i] = self.t
                    self.key_duration_maxes[i] = self.current_key_durations[i]  # set key_duration_max when key released

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

    def collect_input(self, keys):
        self.current_frame = get_frame()
        self.set_current_and_prev_key_state(keys)
        self.set_current_key_scores_and_label(keys)


def append_rectangles_to_image(imagep, scoresp, wasdp, model_output=None):
    width = imagep.shape[1]

    scores = (np.array(scoresp) * 255).astype(np.uint8)
    scores = np.expand_dims(scores, 0)
    scores = cv2.cvtColor(scores, cv2.COLOR_GRAY2RGB)
    scores = cv2.resize(scores, (width, 50), interpolation=cv2.INTER_NEAREST)

    # wasd_row = np.array(wasdp).astype(np.uint8)
    # wasd_row = wasd_row * 255
    # wasd_row = np.expand_dims(wasd_row, 0)
    # wasd_row = cv2.cvtColor(wasd_row, cv2.COLOR_GRAY2RGB)
    # wasd_row = cv2.resize(wasd_row, (width, 50), interpolation=cv2.INTER_NEAREST)

    if model_output is None:
        wasd_bg = np.array(wasdp).astype('uint8') * 255
        wasd_fg = list(map(lambda x: (int(x),), wasdp))  # tuples containing value cast to python int

    else:
        wasd_bg = (model_output * 255).astype('uint8')
        wasd_fg = list(map(lambda x: (int(x),), wasdp))  # tuples containing value cast to python int

    for i, colour in enumerate(wasd_fg):
        if colour[0] > 127:
            wasd_fg[i] = (64,)

    wasd_bg = np.expand_dims(wasd_bg, 0)
    wasd_bg = cv2.cvtColor(wasd_bg, cv2.COLOR_GRAY2RGB)
    wasd_bg = cv2.resize(wasd_bg, (width, 50), interpolation=cv2.INTER_NEAREST)

    text_top_left = (round(width * 0.1), round(50 * 0.9))
    cv2.putText(wasd_bg, 'W', text_top_left, FONT, 2, wasd_fg[0], 3)
    text_top_left = (round(width * 0.35), round(50 * 0.9))
    cv2.putText(wasd_bg, 'A', text_top_left, FONT, 2, wasd_fg[1], 3)
    text_top_left = (round(width * 0.60), round(50 * 0.9))
    cv2.putText(wasd_bg, 'S', text_top_left, FONT, 2, wasd_fg[2], 3)
    text_top_left = (round(width * 0.85), round(50 * 0.9))
    cv2.putText(wasd_bg, 'D', text_top_left, FONT, 2, wasd_fg[3], 3)
    # print(imagep.dtype, scores.dtype, wasd_bg.dtype)
    return np.vstack((imagep, scores, wasd_bg))


def display_frame(q_param):
    while True:
        while True:
            frame, key_scores, key_states = q_param.get()

            if frame is None:
                return
            elif q_param.qsize() == 0:
                break

        frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation=cv2.INTER_NEAREST)
        text_top_left = (round(DISPLAY_WIDTH * 0.1), round(DISPLAY_HEIGHT * 0.9))
        cv2.putText(frame, 'COLLECTING DATA', text_top_left, FONT, 2, (255, 255, 255), 3)

        frame = append_rectangles_to_image(frame, key_scores, key_states)
        cv2.imshow('KEY SCORES AND KEYS PRESSED', frame)

        # waitKey has to be called between imshow calls
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            return


initial_data_fn = 'initial_data.npy'
correction_data_fn = 'correction_data.npy'

IMAGE_SHAPE = (INPUT_HEIGHT, INPUT_WIDTH, 3)
FRAME_SEQ_LENGTH = 2
FRAME_SEQ_SHAPE = (FRAME_SEQ_LENGTH,) + IMAGE_SHAPE

DATUM_DTYPE = np.dtype([('image_seq', np.uint8, FRAME_SEQ_SHAPE), ('key_score', np.float32, (4,)),
                        ('prev_key_state', np.float32, (4,)), ('label', np.float32, (4,))])


class DataCollector(InputCollector):

    def __init__(self, array_length, q_param, correction_data=False):
        super().__init__()

        data = np.zeros(array_length, dtype=DATUM_DTYPE)
        self.data = np.rec.array(data)

        self.frame_seq = np.zeros(FRAME_SEQ_SHAPE, dtype=np.uint8)

        self.array_length = array_length
        self.correction_data = correction_data

        self.index = 0
        # W pressed, W remaining unpressed, W released, W remaining pressed, etc.
        self.signal_counts = [0, 0, 0, 0,   0, 0, 0, 0,   0, 0, 0, 0,   0, 0, 0, 0]

        self.fso = 0
        self.skip_counter = 0
        self.SKIP_COUNTER_MAX = 4

        self.correcting = False

        self.q = q_param
        self.put_counter = 0

    def conditionally_add_datum(self):
        decision = True

        if ((self.previous_key_states == [True, False, False, False])
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
                if not self.previous_key_states[i] and self.current_key_state[i]:  # Key Pressed
                    self.signal_counts[i * 4] += 1
                elif not self.previous_key_states[i] and not self.current_key_state[i]:  # Key Remaining Unpressed
                    self.signal_counts[i * 4 + 1] += 1
                elif self.previous_key_states[i] and not self.current_key_state[i]:  # Key Released
                    self.signal_counts[i * 4 + 2] += 1
                elif self.previous_key_states[i] and self.current_key_state[i]:  # Key Remaining Pressed
                    self.signal_counts[i * 4 + 3] += 1

        if decision:
            self.data[self.index] = (self.frame_seq, self.current_key_scores,
                                     self.previous_key_states, self.current_label)

            self.index += 1

        if (self.index % round(self.array_length/100) == 0) and (self.index != 0):
            self.print_signals()

    def update_frame_seq(self):
        self.frame_seq = np.roll(self.frame_seq, 1, axis=0)
        self.frame_seq[0] = self.current_frame

    def collect_datum(self, keys):
        self.collect_input(keys)
        self.update_frame_seq()
        self.conditionally_add_datum()

        if not self.correction_data:
            self.put_counter = (self.put_counter + 1) % 2
            if self.put_counter == 0:
                self.q.put((self.current_frame, self.current_key_scores, self.current_key_state))

    def save(self):
        if self.correction_data:
            filename = correction_data_fn

            if os.path.isfile(correction_data_fn):
                print('Correction data already exists. Appending timestamp to filenames.')
                append_string = '_' + str(time())
                filename += append_string

        else:
            choice = 'y'
            if os.path.isfile(initial_data_fn):
                choice = input('Overwrite existing initial data? (Y/n)')

            if choice != 'n':
                filename = initial_data_fn
            else:
                return

        print('Saving {} frames'.format(self.index))

        self.data = self.data[:self.index]

        np.save(filename, self.data)

    def stop_session_decision(self):
        # Always increment index *after* deciding whether to save datum
        return self.index == self.array_length

    def print_signals(self):
        print()
        print('fso', self.fso)
        print('index', self.index)
        print(self.signal_counts[0:4])
        print(self.signal_counts[4:8])
        print(self.signal_counts[8:12])
        print(self.signal_counts[12:16])

    def quit_collector(self):
        self.q.put((None, None, None))


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


def imshow(data_param, width=750, height=None, frame_rate=0, title='imshow'):
    if height is None:
        height = data_param[0].image_seq[0].shape[0] * round(width / data_param[0].image_seq[0].shape[1])

    names = data_param.dtype.names

    for datum in data_param:
        st = time()
        for name, field in zip(names, datum):
            if name == 'image_seq':
                print()
            else:
                print(name, field)

        if datum.image_seq.dtype == np.float32:
            image_seq = (datum.image_seq * 255).astype(np.uint8)
        else:
            image_seq = datum.image_seq

        for image in image_seq:
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST)
            image = append_rectangles_to_image(image, datum.key_score, datum.prev_key_state)

            cv2.imshow(title, image)
            cv2.waitKey(25)
            sleep(0.2)

        if frame_rate == 0:
            if cv2.waitKey(0) == ord('q'):
                cv2.destroyAllWindows()
                return
        else:
            if cv2.waitKey(25) == ord('q'):
                cv2.destroyAllWindows()
                return
            sleep(max(0, round(1/frame_rate - (time() - st))))

    cv2.destroyAllWindows()


def check_data(prepare=False, show=True, correction_data=False):
    if correction_data:
        data = np.load(correction_data_fn)
    else:
        data = np.load(initial_data_fn)

    # INCORPORATE CORRECTION DATA?

    data = np.rec.array(data)

    print('data.shape', data.shape)
    print('data.dtype', data.dtype)
    print('data.image_seq.shape', data.image_seq.shape)
    print('data.key_score.shape', data.key_score.shape)
    print('data.prev_key_state.shape', data.prev_key_state.shape)
    print('data.label.shape', data.label.shape)

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

    for image_seq, key_score, prev_key_state, label in data:
        label = list(label.astype(np.bool))
        prev_key_state = list(prev_key_state.astype(np.bool))

        if label == [True, False, False, False] and prev_key_state == [True, False, False, False]:
            forwards_and_straight_on += 1

        if not prev_key_state[0]:
            if label[0]:
                forwards_pressed += 1
            else:
                forwards_remaining_unpressed += 1
        elif prev_key_state[0]:
            if not label[0]:
                forwards_released += 1
            else:
                forwards_remaining_pressed += 1

        if not prev_key_state[1]:
            if label[1]:
                lefts_pressed += 1
            else:
                lefts_remaining_unpressed += 1
        elif prev_key_state[1]:
            if not label[1]:
                lefts_released += 1
            else:
                lefts_remaining_pressed += 1

        if not prev_key_state[2]:
            if label[2]:
                brakes_pressed += 1
            else:
                brakes_remaining_unpressed += 1
        elif prev_key_state[2]:
            if not label[2]:
                brakes_released += 1
            else:
                brakes_remaining_pressed += 1

        if not prev_key_state[3]:
            if label[3]:
                rights_pressed += 1
            else:
                rights_remaining_unpressed += 1
        elif prev_key_state[3]:
            if not label[3]:
                rights_released += 1
            else:
                rights_remaining_pressed += 1

    if prepare:
        prepare_data(data,
                     [forwards_pressed, forwards_remaining_unpressed, forwards_released, forwards_remaining_pressed,
                      lefts_pressed,    lefts_remaining_unpressed,    lefts_released,    lefts_remaining_pressed,
                      brakes_pressed,   brakes_remaining_unpressed,   brakes_released,   brakes_remaining_pressed,
                      rights_pressed,   rights_remaining_unpressed,   rights_released,   rights_remaining_pressed],
                     show=show)
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
            imshow(data)
        else:
            pass


def key_transition_check(key, transition_name, prev_key_state, label):
    i = ['w', 'a', 's', 'd'].index(key)

    if transition_name == 'pressed':
        return (not prev_key_state[i]) and label[i]
    elif transition_name == 'remaining_unpressed':
        return (not prev_key_state[i]) and (not label[i])
    elif transition_name == 'released':
        return prev_key_state[i] and (not label[i])
    elif transition_name == 'remaining_pressed':
        return prev_key_state[i] and label[i]


def add_random_noise_to_float32_array(nparray, amplitude):
    rng = np.random.default_rng().integers

    nparray = (nparray * 255).astype(np.int16)
    noise = rng(- amplitude, amplitude, nparray.size, dtype=np.int16).reshape(nparray.shape)

    nparray = (nparray + noise).astype(np.uint8)
    nparray = nparray.astype(np.float32) / 255

    return nparray


training_data_fn = 'training_data.npy'

wasd = ['w', 'a', 's', 'd']
transitions = ['pressed', 'remaining_unpressed', 'released', 'remaining_pressed']

TRAINING_DTYPE = np.dtype([('image_seq', np.float32, FRAME_SEQ_SHAPE), ('key_score', np.float32, (4,)),
                           ('prev_key_state', np.float32, (4,)), ('label', np.float32, (4,))])


def prepare_data(data_param, stats_list, duplication_factor=4, noise_amplitude=8, show=False):

    training_data_array_length = min(stats_list) * duplication_factor * len(stats_list)  # len(stats_list)==num signals

    training_data = np.zeros(training_data_array_length, dtype=TRAINING_DTYPE)
    training_data = np.rec.array(training_data)

    training_data_index = 0
    # frame_type_accumulator = [0, 0, 0, 0,   0, 0, 0, 0,   0, 0, 0, 0,   0, 0, 0, 0]

    min_stat = min(stats_list)
    min_datum_list_len = min_stat * duplication_factor

    for signal, (key, transition) in enumerate(product(wasd, transitions)):
        datum_list_index = 0
        stat = stats_list[signal]

        datum_list = np.zeros(stat, dtype=TRAINING_DTYPE)
        datum_list = np.rec.array(datum_list)
        print('\ninitial length', datum_list.shape)

        for datum in data_param:
            if key_transition_check(key, transition, datum.prev_key_state, datum.label):
                datum_list[datum_list_index].image_seq = datum.image_seq.astype(np.float32) / 255  # Convert and rescale
                datum_list[datum_list_index].key_score = datum.key_score
                datum_list[datum_list_index].prev_key_state = datum.prev_key_state
                datum_list[datum_list_index].label = datum.label
                datum_list_index += 1

        if stat < min_datum_list_len:
            ceiled_len_ratio = ceil(min_datum_list_len / stat)

            datum_list = np.repeat(datum_list, ceiled_len_ratio)
            print('repeated to', datum_list.shape)
            datum_list = datum_list[:min_datum_list_len]
            datum_list.image_seq = add_random_noise_to_float32_array(datum_list.image_seq, noise_amplitude)
        else:
            jump = floor(stat / min_datum_list_len)

            datum_list = datum_list[::jump]
            print('sliced to', datum_list.shape)
            datum_list = datum_list[:min_datum_list_len]

        print('truncated to', datum_list.shape)
        print('training_data[{}:({} + {})]'.format(training_data_index, training_data_index, min_datum_list_len))
        print(np.max(datum_list.image_seq[0][0]))
        training_data[training_data_index:(training_data_index + min_datum_list_len)] = datum_list
        training_data_index += min_datum_list_len

    print(training_data.image_seq.dtype, np.max(training_data.image_seq[0][0]))

    if show:
        imshow(training_data)

    choice = input('Save? (y/N)')
    if choice == 'y':
        if os.path.isfile(training_data_fn):
            choice = input('Overwrite existing training data? (y/n)')

        if choice == 'y':
            np.save(training_data_fn, training_data)


def train_new_model(epochs=1):
    # Images converted to float32 and rescaled at saving time

    conv_input = ks.Input(shape=(INPUT_HEIGHT, INPUT_WIDTH, 3), name='conv_input')
    x = ks.layers.Conv2D(6, 4, padding='same', activation='relu', data_format='channels_last',
                         name='1st_conv')(conv_input)
    x = ks.layers.MaxPooling2D(data_format='channels_last')(x)
    x = ks.layers.Conv2D(12, 4, padding='same', activation='relu', data_format='channels_last',
                         name='2nd_conv')(x)
    x = ks.layers.MaxPooling2D(data_format='channels_last')(x)
    x = ks.layers.Conv2D(24, 4, padding='same', activation='relu', data_format='channels_last',
                         name='3rd_conv')(x)
    x = ks.layers.MaxPooling2D(data_format='channels_last')(x)
    x = ks.layers.Flatten()(x)
    x = ks.layers.Dense(16, activation='relu', name='image_encoding')(x)

    conv_model = ks.Model(inputs=conv_input, outputs=x)

    recurrent_input_1 = ks.Input(shape=(16,), name='recurrent_input_1')
    recurrent_input_2 = ks.Input(shape=(16,), name='recurrent_input_2')
    y = ks.layers.concatenate([recurrent_input_1, recurrent_input_2])
    y = ks.layers.Reshape((2, 16))(y)
    y = ks.layers.LSTM(16)(y)
    recurrent_model = ks.Model(inputs=[recurrent_input_1, recurrent_input_2], outputs=y)

    image_input_1 = ks.Input(shape=(INPUT_HEIGHT, INPUT_WIDTH, 3), name='image_input_1')
    image_input_2 = ks.Input(shape=(INPUT_HEIGHT, INPUT_WIDTH, 3), name='image_input_2')
    encoded_image_1 = conv_model(image_input_1)
    encoded_image_2 = conv_model(image_input_2)

    z = recurrent_model([encoded_image_1, encoded_image_2])

    im_seq_model = ks.Model(inputs=[image_input_1, image_input_2], outputs=z)

    encoded_input = ks.Input(shape=(16,), name='encoded_input')

    scores_input = ks.Input(shape=(4,), name='scores_input')
    w = ks.layers.Dense(8, activation='relu')(scores_input)

    states_input = ks.Input(shape=(4,), name='states_input')
    v = ks.layers.Dense(8, activation='relu')(states_input)

    u = ks.layers.concatenate([encoded_input, w, v])
    combining_model_output = ks.layers.Dense(4, activation='sigmoid', name='model_output')(u)

    combining_model = ks.Model(inputs=[encoded_input, scores_input, states_input], outputs=combining_model_output)

    encoded_im_seq = im_seq_model([image_input_1, image_input_2])
    full_model_output = combining_model([encoded_im_seq, scores_input, states_input])

    full_model = ks.Model(inputs=[image_input_1, image_input_2, scores_input, states_input], outputs=full_model_output)

    print(combining_model.summary())
    print(full_model.summary())

    data = np.load(training_data_fn, mmap_mode='r')
    data = np.rec.array(data)

    full_model.compile(loss=ks.losses.MeanSquaredError(), metrics=['accuracy'])

    full_model.fit({'image_input_1': data.image_seq[:, 0], 'image_input_2': data.image_seq[:, 1],
                   'scores_input': data.key_score, 'states_input': data.prev_key_state},
                   {'model_output': data.label},
                   epochs=epochs, shuffle=True, validation_split=0.2)

    while True:
        choice = input('Enter a number N to continue training for N epochs\n'
                       'Enter s to save and quit\n'
                       'Enter c to save the model and train for another epoch\n'
                       'Enter q to quit without saving\n'
                       '(N/s/q)\n')

        if choice.isnumeric():
            full_model.fit({'image_input': data.image, 'scores_input': data.key_score,
                           'states_input': data.prev_key_states}, {'model_output': data.label},
                           epochs=int(choice), shuffle=True, validation_split=0.2)
        elif choice == 's':
            full_model.save(MODEL_NAME)
            return
        elif choice == 'c':
            full_model.save(MODEL_NAME)
            full_model.fit({'image_input': data.image, 'scores_input': data.key_score,
                           'states_input': data.prev_key_states}, {'model_output': data.label},
                           epochs=1, shuffle=True, validation_split=0.2)
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


# import warnings
# warnings.simplefilter('default')
# warnings.simplefilter('error')
# try:
#     np_array = (np_array - array_min) / (array_max - array_min)
# except RuntimeWarning:
#     print('RuntimeWarning', np_array.shape)


def rescale(np_array):
    array_max = np.max((0.1, np.max(np_array)))  # stops zero denom below
    array_min = np.min(np_array)

    np_array = (np_array - array_min) / (array_max - array_min)

    return np_array


def display_features(q_param):
    while True:
        # start_time = time()
        # sleep(1./15)
        while True:
            current_key_state, current_key_durations, model_layers, correcting = q_param.get()
            if q_param.qsize() == 0:
                break
                # sleep(1)
            else:
                pass

        if current_key_state is None:
            print('Returning subprocess')
            break
        else:
            pass

        conv1 = rescale(model_layers[0][0].numpy())  # conv1.shape == (height, width, num_conv_filters)
        conv1 = np.transpose(conv1, axes=(2, 0, 1))  # 2nd dim of input goes to 0th dim of output etc
        # could use dsplit here

        conv2 = rescale(model_layers[1][0].numpy())
        conv2 = np.transpose(conv2, axes=(2, 0, 1))
        conv2 = conv2[:6]  # first six filters from second conv layer

        conv_img1 = np.vstack(conv1)
        conv_img1 = cv2.resize(conv_img1, (320, 900), interpolation=cv2.INTER_NEAREST)

        conv_img2 = np.vstack(conv2)
        conv_img2 = cv2.resize(conv_img2, (320, 900), interpolation=cv2.INTER_NEAREST)

        conv_features = np.hstack((conv_img1, conv_img2))

        if correcting:
            # 1.0 is white because conv_features is still float32
            cv2.putText(conv_features, 'Correcting', (128, 240), FONT, 2, (1.0,), 3)
            cv2.putText(conv_features, 'Correcting', (128, 600), FONT, 2, (1.0,), 3)
            cv2.putText(conv_features, 'Correcting', (128, 960), FONT, 2, (1.0,), 3)

        conv_features = (conv_features * 255).astype('uint8')

        penult_dense = rescale(model_layers[2][0].numpy())
        penult_dense = (penult_dense * 255).astype('uint8')
        penult_dense = np.expand_dims(penult_dense, 0)
        penult_dense = cv2.resize(penult_dense, (640, 30), interpolation=cv2.INTER_NEAREST)

        features_scores_keys = np.vstack((conv_features, penult_dense))
        features_scores_keys = cv2.cvtColor(features_scores_keys, cv2.COLOR_GRAY2RGB)
        features_scores_keys = append_rectangles_to_image(features_scores_keys,
                                                          current_key_durations, current_key_state,
                                                          model_output=model_layers[-1][0].numpy())

        cv2.imshow('FEATURES, SCORES and KEYS PRESSED', features_scores_keys)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            return  # pressing q on the imshow window doesn't cause the subprocess to stop. don't know why
        #
        # # print(round(1 / (time() - start_time)))


layer_outputs = [0, 0, 0, 0]  # PyCharm appeasement


class ModelRunner(InputCollector):
    def __init__(self, q_param, correction_data=False):
        super().__init__()
        # from tensorflow.keras.models import load_model, Model
        model = ks.models.load_model(MODEL_NAME)
        model = ks.models.Model(inputs=model.inputs, outputs=[model.get_layer('1st_conv').output,
                                                              model.get_layer('2nd_conv').output,
                                                              model.get_layer('penult_dense').output,
                                                              model.get_layer('model_output').output])
        self.model = model

        self.correction_data = correction_data
        self.stuck = False
        self.stuck_time = 0
        self.dormant = False
        self.most_recent_not_coasting = time()

        self.q = q_param
        self.q_counter = 0

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

        # Annoying
        # if (not correcting) and self.current_key_durations[0] > 15.0:
        #     # self.current_key_durations[0] = 0
        #     # self.key_duration_mins[0] = 0
        #     # self.key_press_times[0] = self.t
        #     # self.key_duration_maxes[0] = 0
        #     if not self.stuck:
        #         print('reversing')
        #
        #     self.stuck = True
        #     self.stuck_time = self.t
        #     PressKey(K)
        # elif self.stuck and ((self.t - self.stuck_time) > 2.9):
        #     self.stuck = False
        #     ReleaseKey(K)

        if (not correcting) or self.correction_data:
            layer_outputs = self.model([np.expand_dims(self.current_frame.astype('float32') / 255, 0),
                                        np.expand_dims(np.array(self.current_key_scores, dtype='float32'), 0),
                                        np.expand_dims(np.array(self.previous_key_states, dtype='float32'), 0)])

        if not correcting:
            prediction_and_correction_to_key_presses(layer_outputs[-1][0], keys, self.dormant)
        elif self.correction_data:
            prediction_and_correction_to_key_presses(layer_outputs[-1][0], keys, self.dormant)

        self.q_counter = (self.q_counter + 1) % 4
        if self.q_counter == 0:
            self.q.put((self.current_key_state, self.current_key_durations, layer_outputs, correcting))

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
                    PressKey(I)
                elif (dormant_time >= 1.0) and ('I' in keys):
                    ReleaseKey(I)

    def quit_model(self):
        self.q.put((None, None, None, None))


def prediction_and_correction_to_key_presses(pred, keys, dormant):
    threshold = 0.5
    # opp_keys = ['K', 'L', 'I', 'J']  # opposite keys
    # don't press left or right if I'm pressing forward

    forward_but_not_dormant = (FORWARD not in keys) and not dormant  # so the model can steer during dormant state

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
