import numpy as np

from common import INITIAL_DATA_FILE_NAME, OUTPUT_SHAPE

data_uint8 = np.load(INITIAL_DATA_FILE_NAME)

TRAINING_DATA_PROPORTION = 0.8

f_count = 0
c_count = 0
b_count = 0
l_count = 0
so_count = 0
r_count = 0

np.random.shuffle(data_uint8)
data_float32 = data_uint8.astype('float32')
print(data_float32.shape)

for datum in data_float32:
    output = datum[0, -1, :OUTPUT_SHAPE[0]].astype('bool')

    if output[0, 0]:
        f_count += 1
    elif output[0, 1]:
        c_count += 1
    elif output[0, 2]:
        b_count += 1
    else:
        print('huh?')
        print(output)
        exit()

    if output[1, 0]:
        l_count += 1
    elif output[1, 1]:
        so_count += 1
    elif output[1, 2]:
        r_count += 1
    else:
        print('huh?')
        print(output)
        exit()

print('f_count {}, c_count {}, b_count {}'.format(f_count, c_count, b_count))
print('l_count {}, so_count {}, r_count {}'.format(l_count, so_count, r_count))

test_split_index = round(len(data_float32) * TRAINING_DATA_PROPORTION)

training_data = data_float32[:test_split_index]
test_data = data_float32[test_split_index:]

training_images = training_data[:, :, :-1] / 255.0  # downscale images but not labels
training_labels = training_data[:, 0, -1, :OUTPUT_SHAPE[0]]

test_images = test_data[:, :, :-1] / 255.0
test_labels = test_data[:, 0, -1, :OUTPUT_SHAPE[0]]

print('train imgs {}, train labels {}'.format(training_images.shape,
                                              training_labels.shape,))

print('test imgs {}, test labels {}'.format(test_images.shape,
                                            test_labels.shape))

choice = input('Check data? (y/n)\n')
if choice == 'y':
    from check_oh6_data import check
    check(data_uint8)

choice = input('Save data? (y/n)\n')
if choice == 'y':
    np.save('training_images.npy', training_images)
    np.save('training_labels.npy', training_labels)
    np.save('test_images.npy', test_images)
    np.save('test_labels.npy', test_labels)
