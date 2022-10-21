from operator import add
from itertools import repeat
from math import ceil, floor
import os

import numpy as np
import tensorflow.keras as ks

from common import imshow, INPUT_WIDTH, INPUT_HEIGHT
from collect_initial import images_fn, key_scores_fn, prev_key_states_fn, labels_fn, OUTPUT_LENGTH
from . import MODEL_PATH

correction_images_fn = 'correction_images_uint8.npy'
correction_key_scores_fn = 'correction_key_scores_float32.npy'
correction_prev_key_states_fn = 'correction_prev_key_states_float32.npy'
correction_labels_fn = 'correction_labels_float32.npy'


def check_data(return_stuff=False):  # , show=True):
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

        # if show:
        #     imshow(images, labels={'key score': key_scores, 'key state': prev_key_states, 'label': labels})
        # else:
        #     pass


training_images_fn = 'training_images_float32.npy'
training_scores_fn = 'training_scores.npy'
training_prev_states_fn = 'training_prev_states.npy'
training_labels_fn = 'training_labels.npy'


def prepare_data(duplication_factor=4, noise_amplitude=8):  # , show=False):
    # shuffling done with tensorflow

    images, key_scores, prev_key_states_bools, labels_bools, stats_list = check_data(return_stuff=True)

    if os.path.isfile(correction_images_fn):
        choice = input('Incorporate correction data? (y/N)')
    else:
        choice = 'n'

    if choice == 'y':
        # print('Tell check_data to load correction data this time')
        (corr_images, corr_key_scores, corr_prev_key_states_bools, corr_labels_bools,
         corr_stats_list) = check_data(return_stuff=True)

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

    # if show:
    #     imshow((images * 255).astype('uint8'), labels={'key score': key_scores, 'key state': prev_key_states,
    #                                                    'label': labels, 'index': range(len(labels))})

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

