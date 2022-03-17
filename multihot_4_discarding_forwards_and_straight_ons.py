import numpy as np
import cv2

from common import (INPUT_WIDTH, INPUT_HEIGHT, get_gta_window, FORWARD, LEFT,
                    BRAKE, RIGHT, release_keys, PressKey, W, S, A, D)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# Model Format:
# Two TensorFlow models which both take the game window as input and output a single value. The two values respectively
# represent coasting or braking, and turning left or turning right. Frames on which I am accelerating and not turning
# are discarded. Using two models allows me to collect more turning frames because I don't have to worry about coasting
# and braking also being balanced within that data.

DATUM_SHAPE = (INPUT_HEIGHT + 1, INPUT_WIDTH, 3)
OUTPUT_SHAPE = (4,)

# coasting, braking, turning left, turning right
output_counts = [0, 0, 0, 0]


# Number of distinct signals which the model can output. A datum can serve as an example of more than one output signal
# e.g. braking and turning left on the same frame.
def number_of_distinct_signals_to_be_balanced():
    return len(output_counts)


def correction_keys_to_output(keys):
    output = np.zeros(OUTPUT_SHAPE, dtype='bool')

    if FORWARD in keys:
        output[0] = True

    if LEFT in keys:
        output[1] = True

    if BRAKE in keys:
        output[2] = True

    if RIGHT in keys:
        output[3] = True

    return output


def get_frame():
    frame = get_gta_window()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
    frame = cv2.resize(frame, (INPUT_WIDTH, INPUT_HEIGHT))
    return frame


def create_datum(frame, output):
    output_row = np.zeros((1, INPUT_WIDTH, 3), dtype='uint8')
    output_row[0, :4, 0] = output.astype('uint8')

    datum = np.concatenate((frame, output_row))

    return datum


# def prediction_to_key_presses(pred, keys=None):

def prediction_and_correction_to_keypresses(pred, keys):
    lr_pred = pred[0]
    cb_pred = pred[1]
    threshold = 0.85
    print(cb_pred)

    release_keys()
    if ((cb_pred > threshold) and not (FORWARD in keys)) or (BRAKE in keys):
        PressKey(S)
    elif BRAKE not in keys:
        PressKey(W)
    # Comparison sign is the other way around compared to below
    # if cb_pred < 1 - threshold then the model saying "coast".
    # elif ((cb_pred > 1 - threshold) and not (BRAKE in keys)) or (FORWARD in keys):
    #     PressKey(W)

    # If the prediction is saying "turn left" and I am *not* saying "turn right" then turn left...
    if ((lr_pred > threshold) and not (RIGHT in keys)) or (LEFT in keys):
        PressKey(A)
    elif ((lr_pred < 1 - threshold) and not (LEFT in keys)) or (RIGHT in keys):
        PressKey(D)


# index_increment always being set to 1 instead of being incremented ensures that a datum on
# which I am both braking and turning left, for example, does not cause two increments of data_buffer_index (double
# increments would cause the training data array to have gaps in it).
def save_datum_decision(keys, output_counts_param, data_buffer_index):
    decision = False
    index_increment = 0

    if BRAKE in keys:
        output_counts_param[0] += 1
        decision = True
        index_increment = 1
    # Limit number of coastings to number of brakings
    elif (not (FORWARD in keys)) and (output_counts_param[1] < output_counts_param[0]):
        output_counts_param[1] += 1
        decision = True
        index_increment = 1

    if LEFT in keys:
        output_counts_param[2] += 1
        decision = True
        index_increment = 1
    elif RIGHT in keys:
        output_counts_param[3] += 1
        decision = True
        index_increment = 1

    data_buffer_index += index_increment

    return decision, output_counts_param, data_buffer_index


def output_row_to_wasd_string(outrow):
    output = outrow[:4, 0].astype('bool')

    if output[0]:
        s = 'W'
    else:
        s = ' '

    if output[1]:
        s += 'A'
    else:
        s += ' '

    if output[2]:
        s += 'S'
    else:
        s += ' '

    if output[3]:
        s += 'D'
    else:
        s += ' '

    if s == 'W   ':
        s = 'PROBLEM: s == \'W   \''

    return s


def output_count_summary(training_data):
    lefts = 0
    rights = 0
    brakes = 0
    coasts = 0
    for datum in training_data:
        output = datum[INPUT_HEIGHT, :4, 0].astype('bool')

        if output[2]:
            brakes += 1
        elif not output[0]:
            coasts += 1

        if output[1]:
            lefts += 1

        if output[3]:
            rights += 1

    s = 'coasts: ' + str(coasts) + ', brakes: ' + str(brakes) + ', lefts: ' + str(lefts) + ', rights: ' + str(rights)

    return s, [coasts, brakes, lefts, rights]


def left_rights(training_data, output_counts_param):
    left_count = output_counts_param[2]
    right_count = output_counts_param[3]

    lefts = np.zeros((left_count,) + DATUM_SHAPE, dtype='uint8')
    rights = np.zeros((right_count,) + DATUM_SHAPE, dtype='uint8')

    lefts_index = 0
    rights_index = 0

    for datum in training_data:
        output = datum[INPUT_HEIGHT, :4, 0].astype('bool')

        if output[1]:
            lefts[lefts_index] = datum
            lefts_index += 1

        if output[3]:
            rights[rights_index] = datum
            rights_index += 1

    lr_min = min(lefts_index, rights_index)

    lefts = lefts[:lr_min]
    rights = rights[:lr_min]

    lefts_and_rights = np.concatenate((lefts, rights))

    return lefts_and_rights


def coast_brakes(training_data, output_counts_param):
    coast_count = output_counts_param[0]
    brake_count = output_counts_param[1]

    coasts = np.zeros((coast_count,) + DATUM_SHAPE, dtype='uint8')
    brakes = np.zeros((brake_count,) + DATUM_SHAPE, dtype='uint8')

    coasts_index = 0
    brakes_index = 0

    for datum in training_data:
        output = datum[INPUT_HEIGHT, :4, 0].astype('bool')

        if output[2]:
            brakes[brakes_index] = datum
            brakes_index += 1
        elif not output[0]:
            coasts[coasts_index] = datum
            coasts_index += 1

    cb_min = min(coasts_index, brakes_index)

    coasts = coasts[:cb_min]
    brakes = brakes[:cb_min]

    coasts_and_brakes = np.concatenate((coasts, brakes))

    return coasts_and_brakes


def split_left_rights_into_inputs_and_labels(training_data):
    np.random.shuffle(training_data)

    images_uint8 = training_data[:, :-1]
    print(images_uint8.shape)

    outputs = training_data[:, -1, :4, 0]
    labels_float32 = outputs[:, 1].astype('float32')
    print(labels_float32.shape)

    return images_uint8, labels_float32


def split_coast_brakes_into_inputs_and_labels(training_data):
    np.random.shuffle(training_data)

    images_uint8 = training_data[:, :-1]
    print(images_uint8.shape)

    outputs = training_data[:, -1, :4, 0]
    labels_float32 = outputs[:, 2].astype('float32')
    print(labels_float32.shape)

    return images_uint8, labels_float32


def get_model_definition():
    import tensorflow.keras as ks

    # Rescaling layer rescales and outputs floats
    image_input = ks.Input(shape=(INPUT_HEIGHT, INPUT_WIDTH, 3), name='image_input')
    x = ks.layers.Rescaling(scale=1. / 255)(image_input)
    feature_output = ks.layers.Conv2D(1, 4, padding='same', activation='relu', data_format='channels_last')(x)

    feature_model = ks.Model(inputs=image_input, outputs=feature_output)

    condensing_model = ks.layers.MaxPooling2D(data_format='channels_last')(feature_output)
    # x = ks.layers.Conv2D(6, 4, padding='same', activation='relu', data_format='channels_last')(x)
    # x = ks.layers.MaxPooling2D(data_format='channels_last')(x)
    x = ks.layers.Flatten()(condensing_model)
    x = ks.layers.Dense(16, activation='relu')(x)
    model_output = ks.layers.Dense(1, activation='sigmoid', name='model_output')(x)

    full_model = ks.Model(inputs=image_input, outputs=model_output)

    return full_model, feature_model


def display_feature_frame(window_name, feature_frame):
    features_max = np.max(feature_frame)
    features_min = np.min(feature_frame)

    feature_frame = (feature_frame - features_min) / (features_max - features_min)

    feature_frame = cv2.resize(feature_frame, (960, 540), interpolation=cv2.INTER_NEAREST)
    cv2.imshow(window_name, feature_frame)

    # waitKey has to be called between imshow calls
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        return
