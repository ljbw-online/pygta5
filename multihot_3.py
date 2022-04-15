import numpy as np
import cv2
from time import time

from common import (INPUT_WIDTH, INPUT_HEIGHT, get_gta_window, LEFT,
                    BRAKE, RIGHT, release_keys, PressKey, W, S, A, D, ReleaseKey, KEYS)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
np.set_printoptions(precision=3, floatmode='fixed', suppress=True)  # suppress stops scientific notation

# Model Input: the game image and a four-element array recording the durations for which each of W, A, S and D have
# been pressed for in the recent past. Each element of this array, except the first, is incremented by one point per
# second whilst the key is pressed and decremented at the same rate whilst the key is not pressed, stopping at zero.
# The first element of the array, recording how long W has been pressed for, is reset to zero whenever S is released.
# This stops the first score from continually increasing.

# Model Output: a multi-hot array of length three representing the scores for A, S and D. If the score for a key goes
# above a certain threshold then the key is pressed. W is pressed down by default and is released whenever S is
# pressed.

MODEL_NAME = 'multihot_3'

# DATUM_SHAPE = (INPUT_HEIGHT + 1, INPUT_WIDTH, 3)
IMAGE_SHAPE = (INPUT_HEIGHT, INPUT_WIDTH, 3)

# A, S, D; multi-hot
OUTPUT_LENGTH = 3
OUTPUT_SHAPE = (OUTPUT_LENGTH,)

DATA_TARGET = 0

# Number of distinct signals which the model can output. A datum can serve as an example of more than one output
# signal e.g. braking and turning left on the same frame.
# turning left, braking and possibly but not necessarily turning, turning right, neither turning nor braking
NUM_SIGNALS = 4
MAX_DATA_PER_SIGNAL = 0  # Defining at module level to appease PyCharm, which also doesn't like it being None.

DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 360


class SignalCounts:
    forward_and_straight_on = 0
    left = 0
    brake = 0
    right = 0

    @classmethod
    def list(cls):
        return [cls.forward_and_straight_on, cls.left, cls.brake, cls.right]


def get_frame():
    frame = get_gta_window()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
    frame = cv2.resize(frame, (INPUT_WIDTH, INPUT_HEIGHT))
    return frame


class InputCollector:

    def __init__(self):
        self.current_frame = get_frame()
        self.current_key_scores = [0., 0., 0., 0.]
        self.current_label = [False, False, False]

        self.key_states = [False, False, False, False]
        self.key_press_times = [0., 0., 0., 0.]
        self.key_release_times = [0., 0., 0., 0.]
        self.key_score_maxes = [0., 0., 0., 0.]
        self.key_score_mins = [0., 0., 0., 0.]

    def set_current_label(self, keys):
        self.current_label[0] = LEFT in keys
        self.current_label[1] = BRAKE in keys
        self.current_label[2] = RIGHT in keys

    def set_current_key_scores(self, keys):
        t = time()

        for i in range(4):
            # If a key is down but was not down on the previous frame then it has been pressed, in which case update
            # the key_press_time for that key.
            if KEYS[i] in keys and not self.key_states[i]:
                self.key_states[i] = True
                self.key_press_times[i] = t
                self.key_score_mins[i] = self.current_key_scores[i]
                # Release W when we press S
                if i == 2:
                    self.key_states[0] = False
                    self.key_release_times[0] = t
            elif KEYS[i] not in keys and self.key_states[i]:
                self.key_states[i] = False
                self.key_release_times[i] = t
                # Set the key_score_max when we release the key
                self.key_score_maxes[i] = self.current_key_scores[i]
                # Releasing S has to look like a press of W so that the forward score starts again from zero
                if i == 2:
                    self.key_press_times[0] = t
                    self.key_score_mins[0] = 0

        for i in range(4):
            if self.key_states[i]:
                self.current_key_scores[i] = self.key_score_mins[i] + (t - self.key_press_times[i])
            else:
                self.current_key_scores[i] = max(0., self.key_score_maxes[i] - (t - self.key_release_times[i]))
                # Subtract the time since the key was released from the maximum value that the score reached.

    def collect_input(self, keys):
        self.set_current_label(keys)
        self.current_frame = get_frame()
        self.set_current_key_scores(keys)


class DataCollector(InputCollector):

    def __init__(self, data_target):
        super().__init__()

        global DATA_TARGET
        global MAX_DATA_PER_SIGNAL
        DATA_TARGET = data_target
        MAX_DATA_PER_SIGNAL = data_target / NUM_SIGNALS

        self.correction_data = False

        self.images = np.zeros((data_target,) + IMAGE_SHAPE, dtype='uint8')
        self.key_scores = np.zeros((data_target, 4), dtype='float32')
        self.labels = np.zeros((data_target,) + OUTPUT_SHAPE)

        self.index = 0
        self.signal_counts = SignalCounts

    def conditionally_add_datum(self):
        left = self.current_label[0]
        brake = self.current_label[1]
        right = self.current_label[2]

        decision = False

        if brake and (self.signal_counts.brake < MAX_DATA_PER_SIGNAL):
            decision = True
            self.signal_counts.brake += 1
        elif left and (self.signal_counts.left < MAX_DATA_PER_SIGNAL):
            decision = True
            self.signal_counts.left += 1
        elif right and (self.signal_counts.right < MAX_DATA_PER_SIGNAL):
            decision = True
            self.signal_counts.right += 1
        elif not any(self.current_label) and (self.signal_counts.forward_and_straight_on < MAX_DATA_PER_SIGNAL):
            decision = True
            self.signal_counts.forward_and_straight_on += 1

        if decision:
            self.images[self.index] = self.current_frame
            self.key_scores[self.index] = self.current_key_scores
            self.labels[self.index] = self.current_label

            if (self.index % round(MAX_DATA_PER_SIGNAL/10) == 0) and (self.index != 0):
                print(self.signal_counts.list())

            self.index += 1

    def collect_datum(self, keys):
        self.collect_input(keys)
        self.conditionally_add_datum()
        # if not correction_data:
        if not self.correction_data:
            self.display_frame()

    def display_frame(self):
        display_frame = cv2.resize(self.current_frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT),
                                   interpolation=cv2.INTER_NEAREST)
        text_top_left = (round(DISPLAY_WIDTH * 0.1), round(DISPLAY_HEIGHT * 0.9))
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(display_frame, 'Collecting data', text_top_left, font, 2, (255, 255, 255), 3)
        cv2.imshow('ALANN', display_frame)

        scores = (np.array(self.current_key_scores) * 255)
        for i in range(4):
            scores[i] = min(255., scores[i])
        scores = scores.astype('uint8')
        scores = np.expand_dims(scores, 0)
        scores = cv2.resize(scores, (640, 50), interpolation=cv2.INTER_NEAREST)

        wasd = np.array(self.key_states).astype('uint8')
        wasd = wasd * 255
        wasd = np.expand_dims(wasd, 0)
        wasd = cv2.resize(wasd, (640, 50), interpolation=cv2.INTER_NEAREST)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_top_left = (round(640 * 0.1), round(50 * 0.9))
        cv2.putText(wasd, 'W', text_top_left, font, 2, (128,), 3)
        text_top_left = (round(640 * 0.35), round(50 * 0.9))
        cv2.putText(wasd, 'A', text_top_left, font, 2, (128,), 3)
        text_top_left = (round(640 * 0.60), round(50 * 0.9))
        cv2.putText(wasd, 'S', text_top_left, font, 2, (128,), 3)
        text_top_left = (round(640 * 0.85), round(50 * 0.9))
        cv2.putText(wasd, 'D', text_top_left, font, 2, (128,), 3)

        scores_and_wasd = np.concatenate((scores, wasd))
        cv2.imshow('KEY SCORES AND KEYS PRESSED', scores_and_wasd)

        # waitKey has to be called between imshow calls
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            return

    def save(self):
        if self.correction_data:
            images_fn = 'correction_images_uint8.npy'
            key_scores_fn = 'correction_key_scores_float32.npy'
            labels_fn = 'correction_labels_float32.npy'
        else:
            images_fn = 'images_uint8.npy'
            key_scores_fn = 'key_scores_float32.npy'
            labels_fn = 'labels_float32.npy'

        print('Saving {} frames'.format(self.index))

        self.images = self.images[:self.index]
        np.save(images_fn, self.images)

        self.key_scores = self.key_scores[:self.index]
        np.save(key_scores_fn, self.key_scores)

        self.labels = self.labels[:self.index]
        np.save(labels_fn, self.labels.astype('float32'))

    def stop_session_decision(self):
        # Always increment index *after* deciding whether to save datum
        return self.index == DATA_TARGET


def correction_to_keypresses(keys):
    # release_keys(keys)
    if 'W' not in keys and BRAKE not in keys:
        PressKey(W)

    if LEFT in keys and 'A' not in keys:
        PressKey(A)
    elif LEFT not in keys and 'A' in keys:
        ReleaseKey(A)

    if BRAKE in keys and 'S' not in keys:
        ReleaseKey(W)
        PressKey(S)
    elif BRAKE not in keys and 'S' in keys:
        PressKey(W)
        ReleaseKey(S)

    if RIGHT in keys and 'D' not in keys:
        PressKey(D)
    elif RIGHT not in keys and 'D' in keys:
        ReleaseKey(D)


def check_data():
    if os.path.isfile('correction_images_uint8.npy'):
        choice = input('Check correction data? (y/N)')
        if choice == 'y':
            images = np.load('correction_images_uint8.npy')
            key_scores = np.load('correction_key_scores_float32.npy')
            labels = np.load('correction_labels_float32.npy')
        else:
            images = np.load('images_uint8.npy')
            key_scores = np.load('key_scores_float32.npy')
            labels = np.load('labels_float32.npy')
    else:
        images = np.load('images_uint8.npy')
        key_scores = np.load('key_scores_float32.npy')
        labels = np.load('labels_float32.npy')

    print('images.shape', images.shape)
    print('key_scores.shape', key_scores.shape)
    print('labels.shape', labels.shape)

    forwards = 0
    lefts = 0
    brakes = 0
    rights = 0

    for label in labels:
        label = label.astype('bool')
        if label[0]:
            lefts += 1
        if label[1]:
            brakes += 1
        if label[2]:
            rights += 1
        if not any(label):
            forwards += 1

    print('forwards:', forwards, 'lefts:', lefts, 'brakes:', brakes, 'rights:', rights)

    for image, score, label in zip(images, key_scores, labels):
        print(score, label)
        image = cv2.resize(image, (640, 360), interpolation=cv2.INTER_NEAREST)
        cv2.imshow('images', image)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

    cv2.destroyAllWindows()  # In case we reach the end of a small data set


def prepare_data():
    # rescaling and shuffling now done with tensorflow

    if os.path.isfile('correction_images_uint8.npy'):
        choice = input('Include correction data? (y/N)')
        if choice == 'y':
            initial_images = np.load('images_uint8.npy')
            initial_scores = np.load('key_scores_float32.npy')
            initial_labels = np.load('labels_float32.npy')

            correction_images = np.load('correction_images_uint8.npy')
            correction_scores = np.load('correction_key_scores_float32.npy')
            correction_labels = np.load('correction_labels_float32.npy')

            images = np.concatenate((initial_images, correction_images))
            scores = np.concatenate((initial_scores, correction_scores))
            labels = np.concatenate((initial_labels, correction_labels))
        else:
            images = np.load('images_uint8.npy')
            scores = np.load('key_scores_float32.npy')
            labels = np.load('labels_float32.npy')
    else:
        images = np.load('images_uint8.npy')
        scores = np.load('key_scores_float32.npy')
        labels = np.load('labels_float32.npy')

    # test_split_index = round(len(labels) * 0.8)

    # training_images = images[:test_split_index]
    # training_scores = scores[:test_split_index]
    # training_labels = labels[:test_split_index]
    #
    # test_images = images[test_split_index:]
    # test_scores = scores[test_split_index:]
    # test_labels = labels[test_split_index:]

    print('training_images.shape', images.shape)
    print('training_scores.shape', scores.shape)
    print('training_labels.shape', labels.shape)

    # print('test_images.shape', test_images.shape)
    # print('test_scores.shape', test_scores.shape)
    # print('test_labels.shape', test_labels.shape)

    print('data types:', images.dtype, scores.dtype, labels.dtype)  # , test_images.dtype,
    # test_scores.dtype, test_labels.dtype)

    choice = input('Save data? (y/n)\n')
    if choice == 'y':
        np.save('training_images.npy', images)
        np.save('training_scores.npy', scores)
        np.save('training_labels.npy', labels)

        # np.save('test_images.npy', test_images)
        # np.save('test_scores.npy', test_scores)
        # np.save('test_labels.npy', test_labels)


def train_new_model(epochs=1, model_desc=''):
    import tensorflow.keras as ks
    import datetime
    # Rescaling layer rescales and outputs floats
    image_input = ks.Input(shape=(INPUT_HEIGHT, INPUT_WIDTH, 3), name='image_input')
    x = ks.layers.Rescaling(scale=1. / 255)(image_input)
    x = ks.layers.Conv2D(4, 4, padding='same', activation='relu', data_format='channels_last', kernel_regularizer=ks.regularizers.l2(0.0001))(x)
    x = ks.layers.MaxPooling2D(data_format='channels_last')(x)
    x = ks.layers.Conv2D(8, 4, padding='same', activation='relu', data_format='channels_last', kernel_regularizer=ks.regularizers.l2(0.0001))(x)
    x = ks.layers.MaxPooling2D(data_format='channels_last')(x)
    x = ks.layers.Flatten()(x)
    x = ks.layers.Dense(16, activation='tanh', kernel_regularizer=ks.regularizers.l2(0.0001))(x)

    scores_input = ks.Input(shape=(4,), name='scores_input')
    y = ks.layers.Dense(16, activation='tanh', kernel_regularizer=ks.regularizers.l2(0.0001))(scores_input)

    z = ks.layers.concatenate([x, y])
    z = ks.layers.Dropout(0.4)(z)
    model_output = ks.layers.Dense(OUTPUT_LENGTH, activation='sigmoid', name='model_output', kernel_regularizer=ks.regularizers.l2(0.0001))(z)

    model = ks.Model(inputs=[image_input, scores_input], outputs=model_output)

    print(model.summary())

    training_images = np.load('training_images.npy')
    training_scores = np.load('training_scores.npy')
    training_labels = np.load('training_labels.npy')
    # test_images = np.load('test_images.npy')
    # test_scores = np.load('test_scores.npy')
    # test_labels = np.load('test_labels.npy')

    model.compile(loss=ks.losses.MeanSquaredError(), metrics=['accuracy'])

    log_dir = './logs/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + model_desc
    tensorboard = ks.callbacks.TensorBoard(log_dir=log_dir, update_freq='batch')

    model.fit({'image_input': training_images, 'scores_input': training_scores},
              {'model_output': training_labels},
              epochs=epochs,
              shuffle=True,
              callbacks=[tensorboard],
              validation_split=0.2)

    # for prediction, label in zip(model.predict({'image_input': test_images[:15]}), test_labels[:15]):
    #     print(prediction, label)

    while True:
        # test_loss, test_metric = \
        #     model.evaluate({'image_input': test_images, 'scores_input': test_scores}, {'model_output': test_labels})
        #
        # print('Test accuracy:', test_metric)

        continue_choice = input('Enter a number N to continue training for N epochs\n'
                                'Enter s to save and quit\n'
                                'Enter q to quit without saving\n'
                                '(N/s/q)\n')

        if continue_choice.isnumeric():
            # model.fit({'image_input': training_images, 'scores_input': training_scores},
            #           {'model_output': training_labels},
            #           epochs=int(continue_choice),
            #           shuffle=True,
            #           callbacks=[tensorboard])
            print('NOT IMPLEMENTED')
        elif continue_choice == 's':
            model.save(MODEL_NAME)
            return
        elif continue_choice == 'q':
            return


def merge_correction_data():
    ci1 = np.load('correction_images_uint8_1.npy')
    cks1 = np.load('correction_key_scores_float32_1.npy')
    cl1 = np.load('correction_labels_float32_1.npy')

    ci2 = np.load('correction_images_uint8.npy')
    cks2 = np.load('correction_key_scores_float32.npy')
    cl2 = np.load('correction_labels_float32.npy')

    np.save('correction_images_uint8_2.npy', ci2)
    np.save('correction_key_scores_float32_2.npy', cks2)
    np.save('correction_labels_float32_2.npy', cl2)

    ci = np.concatenate((ci1, ci2))
    cks = np.concatenate((cks1, cks2))
    cl = np.concatenate((cl1, cl2))

    np.save('correction_images_uint8.npy', ci)
    np.save('correction_key_scores_float32.npy', cks)
    np.save('correction_labels_float32.npy', cl)


def truncate_correction_data(index):
    # SHUFFLING AND THEN TRUNCATING IS ACTUALLY NOT WHAT I WANTED TO DO
    # All of the forwards frames are collected at the beginning of the session, so truncating the last 7000 frames
    # makes the data set inbalanced.

    bciu = np.load('bad_correction_images_uint8.npy')
    bcksf = np.load('bad_correction_key_scores_float32.npy')
    bclf = np.load('bad_correction_labels_float32.npy')

    # Shuffling each array separately! Images would have been reassigned a random label!
    np.random.shuffle(bciu)
    np.random.shuffle(bcksf)
    np.random.shuffle(bclf)

    bciu = bciu[:index]
    bcksf = bcksf[:index]
    bclf = bclf[:index]

    np.save('correction_images_uint8_1.npy', bciu)
    np.save('correction_key_scores_float32_1.npy', bcksf)
    np.save('correction_labels_float32_1.npy', bclf)


def prediction_to_key_presses(pred):
    threshold = 0.5
    release_keys()
    PressKey(W)

    if (pred[0] > threshold) and (pred[2] < threshold):
        PressKey(A)
    if pred[1] > threshold:
        ReleaseKey(W)
        PressKey(S)
    if (pred[2] > threshold) and (pred[0] < threshold):
        PressKey(D)


def rescale(np_array):
    array_max = np.max(np_array)
    array_min = np.min(np_array)

    np_array = (np_array - array_min) / (array_max - array_min)

    return np_array


layer_outputs = []  # PyCharm appeasement


class ModelRunner(InputCollector):
    def __init__(self):
        super().__init__()
        from tensorflow.keras.models import load_model, Model
        model = load_model(MODEL_NAME)
        model = Model(inputs=model.inputs, outputs=[layer.output for layer in model.layers])
        self.model = model

    def display_features(self, model_layers, correcting):
        font = cv2.FONT_HERSHEY_SIMPLEX

        conv1 = rescale(model_layers[2][0].numpy())
        conv1_f1 = conv1[:, :, 0]
        conv1_f2 = conv1[:, :, 1]
        conv1_f3 = conv1[:, :, 2]
        conv1_f4 = conv1[:, :, 3]

        conv2 = rescale(model_layers[4][0].numpy())
        conv2_f1 = conv2[:, :, 0]
        conv2_f2 = conv2[:, :, 1]
        conv2_f3 = conv2[:, :, 2]
        conv2_f4 = conv2[:, :, 3]
        conv2_f5 = conv2[:, :, 4]
        conv2_f6 = conv2[:, :, 5]
        conv2_f7 = conv2[:, :, 6]
        conv2_f8 = conv2[:, :, 7]

        conv_img1 = np.concatenate((np.concatenate((conv1_f1, conv1_f2), axis=1), np.concatenate((conv1_f3, conv1_f4), axis=1)))
        conv_img2 = np.concatenate((np.concatenate((conv2_f1, conv2_f2), axis=1), np.concatenate((conv2_f3, conv2_f4), axis=1)))
        conv_img3 = np.concatenate((np.concatenate((conv2_f5, conv2_f6), axis=1), np.concatenate((conv2_f7, conv2_f8), axis=1)))

        conv_img2 = cv2.resize(conv_img2, (320, 180), interpolation=cv2.INTER_NEAREST)
        conv_img3 = cv2.resize(conv_img3, (320, 180), interpolation=cv2.INTER_NEAREST)

        conv_features = np.concatenate((conv_img1, conv_img2, conv_img3))
        conv_features = cv2.resize(conv_features, (640, 1080), interpolation=cv2.INTER_NEAREST)

        if correcting:
            cv2.putText(conv_features, 'Correcting', (128, 240), font, 2, (255,), 3)
            cv2.putText(conv_features, 'Correcting', (128, 600), font, 2, (255,), 3)
            cv2.putText(conv_features, 'Correcting', (128, 960), font, 2, (255,), 3)

        cv2.imshow('CONVOLUTIONAL FILTERS', conv_features)

        penult_dense = rescale(model_layers[-2][0].numpy())
        penult_dense = (penult_dense * 255).astype('uint8')
        penult_dense = np.expand_dims(penult_dense, 0)
        penult_dense = cv2.resize(penult_dense, (640, 20), interpolation=cv2.INTER_NEAREST)

        scores = (np.array(self.current_key_scores) * 255)
        for i in range(4):
            scores[i] = min(255., scores[i])
        scores = scores.astype('uint8')
        scores = np.expand_dims(scores, 0)
        scores = cv2.resize(scores, (640, 50), interpolation=cv2.INTER_NEAREST)

        wasd = np.array(self.key_states).astype('uint8')
        wasd = wasd * 255
        wasd = np.expand_dims(wasd, 0)
        wasd = cv2.resize(wasd, (640, 50), interpolation=cv2.INTER_NEAREST)
        text_top_left = (round(640 * 0.1), round(50 * 0.9))
        cv2.putText(wasd, 'W', text_top_left, font, 2, (128,), 3)
        text_top_left = (round(640 * 0.35), round(50 * 0.9))
        cv2.putText(wasd, 'A', text_top_left, font, 2, (128,), 3)
        text_top_left = (round(640 * 0.60), round(50 * 0.9))
        cv2.putText(wasd, 'S', text_top_left, font, 2, (128,), 3)
        text_top_left = (round(640 * 0.85), round(50 * 0.9))
        cv2.putText(wasd, 'D', text_top_left, font, 2, (128,), 3)

        penult_scores_wasd = np.concatenate((penult_dense, scores, wasd))
        cv2.imshow('KEY SCORES AND KEYS PRESSED', penult_scores_wasd)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            return

    def run_model(self, keys, correcting):
        global layer_outputs
        self.collect_input(keys)

        if not correcting:
            layer_outputs = self.model([np.expand_dims(self.current_frame, 0),
                                        np.expand_dims(np.array(self.current_key_scores, dtype='float32'), 0)])

            prediction_to_key_presses(layer_outputs[-1][0])

        self.display_features(layer_outputs, correcting)


def create_datum(frame, output):
    output_row = np.zeros((1, INPUT_WIDTH, 3), dtype='uint8')
    output_row[0, :OUTPUT_SHAPE[0], 0] = output.astype('uint8')

    datum = np.concatenate((frame, output_row))

    return datum


# averaging_period = 8
# predictions = np.zeros((averaging_period,) + OUTPUT_SHAPE)
# THRESHOLD = 0.75
# averages = np.zeros(OUTPUT_SHAPE)


# def prediction_to_key_presses(current_pred, pred_array):
#     release_keys()
#     PressKey(W)
#
#     pred_array = np.roll(pred_array, 1, axis=0)
#     pred_array[0] = current_pred
#
#     averages[0] = np.average(pred_array[:, 0])
#     averages[1] = np.average(pred_array[:, 1])
#     averages[2] = np.average(pred_array[:, 2])
#
#     if (averages[0] > THRESHOLD) and (averages[2] < THRESHOLD):
#         PressKey(A)
#     if averages[1] > THRESHOLD:
#         ReleaseKey(W)
#         PressKey(S)
#     if (averages[2] > THRESHOLD) and (averages[0] < THRESHOLD):
#         PressKey(D)
#
#     return pred_array

# index_increment always being set to 1 instead of being incremented ensures that a datum on
# which I am both braking and turning left, for example, does not cause two increments of data_buffer_index (double
# increments would cause the training data array to have gaps in it).
def save_datum_decision(keys, output_counts_param, data_buffer_index, max_data_per_output):
    decision = False
    index_increment = 0

    if (BRAKE in keys) and (output_counts_param[1] < max_data_per_output):
        decision = True
        index_increment = 1
        output_counts_param[1] += 1
    elif (LEFT in keys) and (output_counts_param[0] < max_data_per_output):
        decision = True
        index_increment = 1
        output_counts_param[0] += 1
    elif (RIGHT in keys) and (output_counts_param[2] < max_data_per_output):
        decision = True
        index_increment = 1
        output_counts_param[2] += 1
    elif ((BRAKE not in keys) and (LEFT not in keys) and (RIGHT not in keys)
          and (output_counts_param[3] < max_data_per_output)):
        decision = True
        index_increment = 1
        output_counts_param[3] += 1

    data_buffer_index += index_increment

    return decision, output_counts_param, data_buffer_index


def stop_session_decision(data_index, data_target, output_counts_param=None):
    if output_counts_param is None:
        pass  # Gets rid of the "unused argument" warning

    return data_index == data_target - 1


def output_row_to_wasd_string(outrow):
    output = outrow[:OUTPUT_SHAPE[0], 0].astype('bool')

    s = 'W'

    if output[0]:
        s += 'A'

    if output[1]:
        s = ''
        if output[0]:
            s = 'A'

        s += 'S'
        # don't need to append 'D' here; it will be appended in the next if statement

    if output[2]:
        s += 'D'

    return s


def output_count_summary(training_data):
    lefts = 0
    brakes = 0
    rights = 0
    nokeys = 0
    for datum in training_data:
        output = datum[INPUT_HEIGHT, :OUTPUT_SHAPE[0], 0].astype('bool')

        if output[0] and not output[1] and not output[2]:
            lefts += 1
        elif output[2] and not output[0] and not output[1]:
            rights += 1
        elif output[1]:
            brakes += 1
        elif not any(output):
            nokeys += 1
        else:
            print('Bad output: {}'.format(output))

    s = 'lefts: ' + str(lefts) + ', brakes: ' + str(brakes) + ', rights: ' + str(rights) + ', nokeys: ' + str(nokeys)

    return s, [lefts, brakes, rights, nokeys]


def split_into_images_and_labels(training_data):
    np.random.shuffle(training_data)

    images_uint8 = training_data[:, :-1]
    print('images_uint8.shape', images_uint8.shape)

    outputs = training_data[:, -1, :OUTPUT_SHAPE[0], 0]
    labels_float32 = outputs.astype('float32')
    print('labels_float32.shape', labels_float32.shape)

    return images_uint8, labels_float32


def get_model_definition():
    import tensorflow.keras as ks

    # Rescaling layer rescales and outputs floats
    image_input = ks.Input(shape=(INPUT_HEIGHT, INPUT_WIDTH, 3), name='image_input')
    x = ks.layers.Rescaling(scale=1. / 255)(image_input)
    x = ks.layers.Conv2D(3, 4, padding='same', activation='relu', data_format='channels_last')(x)
    x = ks.layers.MaxPooling2D((3, 3), data_format='channels_last')(x)
    # x = ks.layers.Conv2D(6, 4, padding='same', activation='relu', data_format='channels_last')(x)
    # x = ks.layers.MaxPooling2D(data_format='channels_last')(x)
    x = ks.layers.Flatten()(x)
    x = ks.layers.Dense(16, activation='relu')(x)
    model_output = ks.layers.Dense(OUTPUT_LENGTH, activation='sigmoid', name='model_output')(x)

    model = ks.Model(inputs=image_input, outputs=model_output)

    return model


# def display_features(layer_outputs, window_name='FEATURES', correcting=False):
#     conv = rescale(layer_outputs[2][0].numpy())
#     conv_f1 = conv[:, :, 0]
#     conv_f1 = cv2.resize(conv_f1, (640, 360), interpolation=cv2.INTER_NEAREST)
#     conv_f2 = conv[:, :, 1]
#     conv_f2 = cv2.resize(conv_f2, (640, 360), interpolation=cv2.INTER_NEAREST)
#     conv_f3 = conv[:, :, 2]
#     conv_f3 = cv2.resize(conv_f3, (640, 360), interpolation=cv2.INTER_NEAREST)
#
#     conv_features = np.concatenate((conv_f1, conv_f2, conv_f3))
#
#     # maxpool = rescale(layer_outputs[3][0].numpy())
#     # maxpool = cv2.resize(maxpool, (640, 480), interpolation=cv2.INTER_NEAREST)
#
#     penult_dense = rescale(layer_outputs[5][0].numpy())
#     penult_dense = np.expand_dims(penult_dense, 0)
#     penult_dense = cv2.resize(penult_dense, (640, 40), interpolation=cv2.INTER_NEAREST)
#
#     final_dense = rescale(layer_outputs[6][0].numpy())
#     final_dense = np.concatenate(([1], final_dense))
#
#     if final_dense[2] > THRESHOLD:
#         final_dense[0] = 0
#
#     final_dense = np.expand_dims(final_dense, 0)
#     final_dense = cv2.resize(final_dense, (640, 50), interpolation=cv2.INTER_NEAREST)
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     text_top_left = (round(640 * 0.1), round(50 * 0.9))
#     cv2.putText(final_dense, 'W', text_top_left, font, 2, (0,), 3)
#     text_top_left = (round(640 * 0.35), round(50 * 0.9))
#     cv2.putText(final_dense, 'A', text_top_left, font, 2, (0,), 3)
#     text_top_left = (round(640 * 0.60), round(50 * 0.9))
#     cv2.putText(final_dense, 'S', text_top_left, font, 2, (0,), 3)
#     text_top_left = (round(640 * 0.85), round(50 * 0.9))
#     cv2.putText(final_dense, 'D', text_top_left, font, 2, (0,), 3)
#
#     final_averages = np.concatenate(([1], averages))
#
#     if final_averages[2] > THRESHOLD:
#         final_averages[0] = 0
#
#     final_averages = np.expand_dims(final_averages, 0)
#     final_averages = cv2.resize(final_averages, (640, 50), interpolation=cv2.INTER_NEAREST)
#     print(averages)
#
#     dense_features = np.concatenate((penult_dense, final_dense, final_averages))
#
#     conv_window_name = 'CONVOLUTIONAL ' + window_name
#     dense_window_name = 'FULLY CONNECTED ' + window_name
#
#     if correcting:
#         cv2.putText(conv_features, 'Correcting', (128, 240), font, 2, (255,), 3)
#
#     cv2.imshow(conv_window_name, conv_features)
#     cv2.imshow(dense_window_name, dense_features)
#
#     # waitKey has to be called between imshow calls
#     if cv2.waitKey(25) & 0xFF == ord('q'):
#         cv2.destroyAllWindows()
#         return
