import numpy as np
from numpy import array_equal as ae
from common import OUTPUT_LENGTH, w, a, s, d, wa, wd, sa, sd, INPUT_WIDTH, INPUT_HEIGHT, MODEL_NAME
import tensorflow as tf

collected_data = np.load('collected_data.npy')

just_imgs_shape = (len(collected_data),INPUT_HEIGHT,INPUT_WIDTH)

fab_count = 0
forward_count = 0
brake_count = 0
forwards = np.zeros_like(collected_data)
brakes = np.zeros_like(collected_data)

lar_count = 0
left_count = 0
right_count = 0
lefts = np.zeros_like(collected_data)
rights = np.zeros_like(collected_data)

for frame in collected_data:
    output = frame[-1,:OUTPUT_LENGTH]
    if ae(output, w):
        forwards[forward_count] = frame
        forward_count += 1
    elif ae(output, a) or ae(output, wa):
        lefts[left_count] = frame
        left_count += 1
    elif ae(output, d) or ae(output, wd):
        rights[right_count] = frame
        right_count += 1
    elif ae(output, s) or ae(output, sa) or ae(output, sd):
        brakes[brake_count] = frame
        brake_count += 1

print('left_count, right_count, 1st for-loop', left_count, right_count)

lefts = lefts[:left_count]
rights = rights[:right_count]

# np.random.shuffle(lefts) # don't need this because rights outnumber lefts
np.random.shuffle(rights)  # get rights from all over data
rights = rights[:left_count]
lefts_and_rights = np.concatenate((lefts, rights))
# np.random.shuffle(lefts_and_rights)  # stop left_and_rights from being all-lefts followed by all-rights

print('forward_count, brake_count, 1s for-loop', forward_count, brake_count)

forwards = forwards[:forward_count]
brakes = brakes[:brake_count]

np.random.shuffle(forwards)
# np.random.shuffle(brakes) # forwards outnumber brakes
forwards = forwards[:brake_count]
forwards_and_brakes = np.concatenate((forwards, brakes))
# np.random.shuffle(forwards_and_brakes) # lars_and_fabs shuffled below

lars_and_fabs = np.concatenate((lefts_and_rights, forwards_and_brakes))
np.random.shuffle(lars_and_fabs)
labels = np.zeros((len(lars_and_fabs), 4), dtype='float32')

forward_count = 0
left_count = 0
right_count = 0
brake_count = 0
for i in range(len(lars_and_fabs)):
    output = lars_and_fabs[i,-1,:OUTPUT_LENGTH]
    if ae(output, w):
        labels[i] = [1,0,0,0]
        forward_count += 1
    elif ae(output, a) or ae(output, wa):
        labels[i] = [0,1,0,0]
        left_count += 1
    elif ae(output, d) or ae(output, wd):
        labels[i] = [0,0,1,0]
        right_count += 1
    elif ae(output, s) or ae(output, sa) or ae(output, sd):
        labels[i] = [0,0,0,1]
        brake_count += 1
    else:
        print(output)
        exit()

print('lc, rc, fc, bc, 2nd for-loop', left_count,right_count, forward_count, brake_count)
print('len(lars), len(fabs), len(lars_and_fabs)', len(lefts_and_rights), len(forwards_and_brakes), len(lars_and_fabs))

lars_and_fabs = lars_and_fabs.astype('float32') / 255.0

test_split_index = round(len(lars_and_fabs) * 0.25)
train_imgs = lars_and_fabs[test_split_index:,:-1]
train_imgs = np.expand_dims(train_imgs, axis=3)  # So that Conv2D will accept it
print('train_imgs.shape', train_imgs.shape)
test_imgs = lars_and_fabs[:test_split_index,:-1]
test_imgs = np.expand_dims(test_imgs, axis=3)

train_labels = labels[test_split_index:]
test_labels = labels[:test_split_index]

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(INPUT_HEIGHT, INPUT_WIDTH, 1)),
                           # kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
    # tf.keras.layers.Dropout(0.5),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
                           # kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
    # tf.keras.layers.Dropout(0.5),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
                           # kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
    # tf.keras.layers.Dropout(0.5),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    # tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(4, activation='softmax'),
])

if __name__ == '__main__':
    metric_name = 'accuracy'
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=[metric_name])

    model.fit(train_imgs, train_labels, epochs=4)

    test_loss, test_metric = model.evaluate(test_imgs, test_labels, verbose=2)
    print('\n{} on test set:'.format(metric_name), test_metric)

    for prediction, label in zip(model.predict(test_imgs[:15]), test_labels[:15]):
        print(list(prediction), label)

    model.save(MODEL_NAME)

# mae = tf.keras.losses.MeanAbsoluteError(tf.keras.losses.Reduction.SUM)
# mae([[0.,0.]],[[5.,1.]]) == 3.0
# Elementwise subtraction followed by sum
