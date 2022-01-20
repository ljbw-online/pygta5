import numpy as np
from numpy import array_equal as ae
from time import sleep
import os

from common import INITIAL_DATA_FILE_NAME, CORRECTION_DATA_FILE_NAME, oh4_w, oh4_wa, oh4_wd, oh4_s

if os.path.isfile(CORRECTION_DATA_FILE_NAME):
    print('Preparing correction data')
    data_uint8 = np.load(CORRECTION_DATA_FILE_NAME)
else:
    data_uint8 = np.load(INITIAL_DATA_FILE_NAME)

OUTPUT_LENGTH = 4
TRAINING_DATA_PROPORTION = 0.8

f_count = 0
l_count = 0
r_count = 0
b_count = 0

np.random.shuffle(data_uint8)
data_float32 = data_uint8.astype('float32')
print(data_float32.shape)

for datum in data_float32:
    output = datum[-1, :OUTPUT_LENGTH]
    if ae(output, oh4_w):
        f_count += 1
    elif ae(output, oh4_wa):
        l_count += 1
    elif ae(output, oh4_wd):
        r_count += 1
    elif ae(output, oh4_s):
        b_count += 1
    else:
        print('huh?')
        print(output)
        exit()

print('f_count {}, l_count {}, r_count {}, b_count {}'.format(f_count, l_count, r_count, b_count))

test_split_index = round(len(data_float32) * TRAINING_DATA_PROPORTION)

training_data = data_float32[:test_split_index]
test_data = data_float32[test_split_index:]

training_images = training_data[:, :-1] / 255.0  # downscale images but not labels
training_labels = training_data[:, -1, :OUTPUT_LENGTH]

test_images = test_data[:, :-1] / 255.0
test_labels = test_data[:, -1, :OUTPUT_LENGTH]

print('train imgs {}, train labels {}'.format(training_images.shape,
                                                                      training_labels.shape,))

print('test imgs {}, test labels {}'.format(test_images.shape,
                                                                   test_labels.shape))

choice = input('Check data? (y/n)\n')
if choice == 'y':
    import cv2
    from common import DISPLAY_WIDTH, DISPLAY_HEIGHT
    for i in range(len(training_images)):
        frame = cv2.resize((training_images[i] * 255).astype('uint8'), (DISPLAY_WIDTH, DISPLAY_HEIGHT),
                           interpolation=cv2.INTER_NEAREST)
        cv2.imshow('Training Data', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        print('Current output', training_labels[i])
        sleep(0.1)
    cv2.destroyAllWindows()

choice = input('Save data? (y/n)\n')
if choice == 'y':
    np.save('training_images.npy', training_images)
    np.save('training_labels.npy', training_labels)
    np.save('test_images.npy', test_images)
    np.save('test_labels.npy', test_labels)
