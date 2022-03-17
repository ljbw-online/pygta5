import numpy as np
import os

from common import INITIAL_DATA_FILE_NAME, CORRECTION_DATA_FILE_NAME

from multihot_3 import (output_count_summary, split_into_images_and_labels)

TRAINING_DATA_PROPORTION = 0.8

data_uint8 = None
if os.path.isfile(CORRECTION_DATA_FILE_NAME):
    choice = input('Include correction data? (y/n)\n')
    if choice == 'y':
        correction_file_name = CORRECTION_DATA_FILE_NAME
        initial = np.load(INITIAL_DATA_FILE_NAME)
        correction = np.load(correction_file_name)
        initial = np.concatenate((initial, correction))

        correction_file_number = 1
        correction_file_name = correction_file_name[:-4] + '_1.npy'
        while True:
            if os.path.isfile(correction_file_name):
                correction = np.load(correction_file_name)
                print('Appending {}'.format(correction_file_name))
                initial = np.concatenate((initial, correction))
                correction_file_number += 1
                correction_file_name = correction_file_name[:-4] + '_' + str(correction_file_number) + '.npy'
            else:
                break

        data_uint8 = initial
        print(data_uint8.shape)
else:
    data_uint8 = np.load(INITIAL_DATA_FILE_NAME)

summary_string, output_counts = output_count_summary(data_uint8)

print(summary_string)

images_uint8, labels_float32 = split_into_images_and_labels(data_uint8)

test_split_index = round(len(data_uint8) * TRAINING_DATA_PROPORTION)

training_images = images_uint8[:test_split_index]
training_labels = labels_float32[:test_split_index]

test_images = images_uint8[test_split_index:]
test_labels = labels_float32[test_split_index:]

# Rescaling currently in network
# training_images = training_data[:, :, :-1] / 255.0  # rescale images but not labels

print('training_images {}, training_labels {}'.format(training_images.shape, training_labels.shape,))

print('test_images {}, test_labels {}'.format(test_images.shape, test_labels.shape))

choice = input('Save data? (y/n)\n')
if choice == 'y':
    np.save('training_images.npy', training_images)
    np.save('training_labels.npy', training_labels)
    np.save('test_images.npy', test_images)
    np.save('test_labels.npy', test_labels)
