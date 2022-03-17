import numpy as np
import cv2

from common import (INPUT_WIDTH, INPUT_HEIGHT, get_gta_window, FORWARD, LEFT,
                    BRAKE, RIGHT, release_keys, PressKey, W, S, A, D, ReleaseKey)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# Model Format:
# Game image as input, three-element multi-hot array as output.
MODEL_NAME = 'multihot_3'

DATUM_SHAPE = (INPUT_HEIGHT + 1, INPUT_WIDTH, 3)

# A, S, D; multi-hot
OUTPUT_LENGTH = 3
OUTPUT_SHAPE = (OUTPUT_LENGTH,)


# Number of distinct signals which the model can output. A datum can serve as an example of more than one output signal
# e.g. braking and turning left on the same frame.
NUM_SIGNALS = 4
# turning left, braking and possibly but not necessarily turning, turning right, neither turning nor braking
output_counts = np.zeros(NUM_SIGNALS)


def correction_keys_to_output(keys):
    output = np.zeros(OUTPUT_SHAPE, dtype='bool')

    if LEFT in keys:
        output[0] = True

    if BRAKE in keys:
        output[1] = True

    if RIGHT in keys:
        output[2] = True

    return output


def get_frame():
    frame = get_gta_window()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
    frame = cv2.resize(frame, (INPUT_WIDTH, INPUT_HEIGHT))
    return frame


def create_datum(frame, output):
    output_row = np.zeros((1, INPUT_WIDTH, 3), dtype='uint8')
    output_row[0, :OUTPUT_SHAPE[0], 0] = output.astype('uint8')

    datum = np.concatenate((frame, output_row))

    return datum


averaging_period = 8
predictions = np.zeros((averaging_period,) + OUTPUT_SHAPE)
THRESHOLD = 0.75
averages = np.zeros(OUTPUT_SHAPE)


def prediction_to_key_presses(current_pred, pred_array):
    release_keys()
    PressKey(W)

    pred_array = np.roll(pred_array, 1, axis=0)
    pred_array[0] = current_pred

    averages[0] = np.average(pred_array[:, 0])
    averages[1] = np.average(pred_array[:, 1])
    averages[2] = np.average(pred_array[:, 2])

    if (averages[0] > THRESHOLD) and (averages[2] < THRESHOLD):
        PressKey(A)
    if averages[1] > THRESHOLD:
        ReleaseKey(W)
        PressKey(S)
    if (averages[2] > THRESHOLD) and (averages[0] < THRESHOLD):
        PressKey(D)

    return pred_array


def correction_to_keypresses(keys):
    release_keys()
    PressKey(W)

    if LEFT in keys:
        PressKey(A)
    if BRAKE in keys:
        ReleaseKey(W)
        PressKey(S)
    if RIGHT in keys:
        PressKey(D)


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


def rescale(np_array):
    array_max = np.max(np_array)
    array_min = np.min(np_array)

    np_array = (np_array - array_min) / (array_max - array_min)

    return np_array


def display_features(layer_outputs, window_name='FEATURES', correcting=False):
    conv = rescale(layer_outputs[2][0].numpy())
    conv_f1 = conv[:, :, 0]
    conv_f1 = cv2.resize(conv_f1, (640, 360), interpolation=cv2.INTER_NEAREST)
    conv_f2 = conv[:, :, 1]
    conv_f2 = cv2.resize(conv_f2, (640, 360), interpolation=cv2.INTER_NEAREST)
    conv_f3 = conv[:, :, 2]
    conv_f3 = cv2.resize(conv_f3, (640, 360), interpolation=cv2.INTER_NEAREST)

    conv_features = np.concatenate((conv_f1, conv_f2, conv_f3))

    # maxpool = rescale(layer_outputs[3][0].numpy())
    # maxpool = cv2.resize(maxpool, (640, 480), interpolation=cv2.INTER_NEAREST)

    penult_dense = rescale(layer_outputs[5][0].numpy())
    penult_dense = np.expand_dims(penult_dense, 0)
    penult_dense = cv2.resize(penult_dense, (640, 40), interpolation=cv2.INTER_NEAREST)

    final_dense = rescale(layer_outputs[6][0].numpy())
    final_dense = np.concatenate(([1], final_dense))

    if final_dense[2] > THRESHOLD:
        final_dense[0] = 0

    final_dense = np.expand_dims(final_dense, 0)
    final_dense = cv2.resize(final_dense, (640, 50), interpolation=cv2.INTER_NEAREST)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_top_left = (round(640 * 0.1), round(50 * 0.9))
    cv2.putText(final_dense, 'W', text_top_left, font, 2, (0,), 3)
    text_top_left = (round(640 * 0.35), round(50 * 0.9))
    cv2.putText(final_dense, 'A', text_top_left, font, 2, (0,), 3)
    text_top_left = (round(640 * 0.60), round(50 * 0.9))
    cv2.putText(final_dense, 'S', text_top_left, font, 2, (0,), 3)
    text_top_left = (round(640 * 0.85), round(50 * 0.9))
    cv2.putText(final_dense, 'D', text_top_left, font, 2, (0,), 3)

    final_averages = np.concatenate(([1], averages))

    if final_averages[2] > THRESHOLD:
        final_averages[0] = 0

    final_averages = np.expand_dims(final_averages, 0)
    final_averages = cv2.resize(final_averages, (640, 50), interpolation=cv2.INTER_NEAREST)
    print(averages)

    dense_features = np.concatenate((penult_dense, final_dense, final_averages))

    conv_window_name = 'CONVOLUTIONAL ' + window_name
    dense_window_name = 'FULLY CONNECTED ' + window_name

    if correcting:
        cv2.putText(conv_features, 'Correcting', (128, 240), font, 2, (255,), 3)

    cv2.imshow(conv_window_name, conv_features)
    cv2.imshow(dense_window_name, dense_features)

    # waitKey has to be called between imshow calls
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        return
