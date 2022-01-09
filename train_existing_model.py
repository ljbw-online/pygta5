import numpy as np
import tensorflow.keras as ks

from common import CORRECTION_FILE_NAME, MODEL_NAME, OUTPUT_LENGTH, oh4_w, oh4_wa, oh4_wd, oh4_s

correction_data = np.load(CORRECTION_FILE_NAME)
model = ks.models.load_model(MODEL_NAME)
ae = np.array_equal

np.random.shuffle(correction_data)

forward_count = 0
left_count = 0
right_count = 0
brake_count = 0
for i in range(len(correction_data)):
    output = correction_data[i, -1, :OUTPUT_LENGTH]
    if ae(output, oh4_w):
        forward_count += 1
    elif ae(output, oh4_wa):
        left_count += 1
    elif ae(output, oh4_wd):
        right_count += 1
    elif ae(output, oh4_s):
        brake_count += 1
    else:
        print(output)
        exit()

print('forwards', forward_count, 'lefts', left_count, 'rights', right_count, 'brakes', brake_count)

test_split_index = round(len(correction_data) * 0.25)
train_imgs = correction_data[test_split_index:, :-1]
train_imgs = np.expand_dims(train_imgs, axis=3)  # So that Conv2D will accept it
print('train_imgs.shape', train_imgs.shape)
train_labels = correction_data[test_split_index:, -1, :OUTPUT_LENGTH]
print('train_labels.shape', train_labels.shape)
print('train_labels[:10]')
print(train_labels[:10])

test_imgs = correction_data[:test_split_index, :-1]
test_imgs = np.expand_dims(test_imgs, axis=3)
test_labels = correction_data[:test_split_index, -1, :OUTPUT_LENGTH]
print('test_imgs.shape', test_imgs.shape)
print('test_labels[:10]')
print(test_labels[:10])

model.fit(train_imgs, train_labels, epochs=3)

while True:
    test_loss, test_metric = model.evaluate(test_imgs, test_labels, verbose=2)
    print('Test accuracy:', test_metric)

    continue_choice = input('Enter a number N to continue training for N epochs\n'
                            'Enter s to save and quit\n'
                            'Enter q to quit without saving\n'
                            '(N/s/q)\n')

    if continue_choice.isnumeric():
        model.fit(train_imgs, train_labels, epochs=int(continue_choice))
    elif continue_choice == 's':
        model.save(MODEL_NAME)
        exit()
    elif continue_choice == 'q':
        exit()
    else:
        print('huh?')
