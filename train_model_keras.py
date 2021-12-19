import numpy as np
from numpy import array_equal as ae
from common import OUTPUT_LENGTH, a, d, wa, wd, sa, sd, INPUT_WIDTH, INPUT_HEIGHT
import tensorflow as tf

collected_data = np.load('collected_data.npy')

just_imgs_shape = (len(collected_data),INPUT_HEIGHT,INPUT_WIDTH)

lar_count = 0
left_count = 0
right_count = 0
lefts = np.zeros_like(collected_data)
rights = np.zeros_like(collected_data)
for frame in collected_data:
    output = frame[-1,:OUTPUT_LENGTH]
    if ae(output, a) or ae(output, wa) or ae(output, sa):
        lefts[left_count] = frame
        left_count += 1
    elif ae(output, d) or ae(output, wd) or ae(output, sd):
        rights[right_count] = frame
        right_count += 1

print('left_count, right_count, 1st for-loop', left_count,right_count)

lefts = lefts[:left_count]
rights = rights[:right_count]

np.random.shuffle(rights)
rights = rights[:left_count]
lefts_and_rights = np.concatenate((lefts,rights))
np.random.shuffle(lefts_and_rights)
labels = np.zeros(len(lefts_and_rights),dtype='float32')

left_count = 0
right_count = 0
for i in range(len(lefts_and_rights)):
    output = lefts_and_rights[i,-1,:OUTPUT_LENGTH]
    if ae(output, a) or ae(output, wa) or ae(output, sa):
        labels[i] = 0
        left_count += 1
    elif ae(output, d) or ae(output, wd) or ae(output, sd):
        labels[i] = 1
        right_count += 1
    else:
        print(output)
        exit()

print('lc, rc, 2nd for-loop', left_count,right_count)
print('len(lars)', len(lefts_and_rights))

lefts_and_rights = lefts_and_rights.astype('float32') / 255.0

test_split_index = round(len(lefts_and_rights) * 0.25)
train_imgs = lefts_and_rights[test_split_index:,:-1]
train_imgs = np.expand_dims(train_imgs, axis=3) # So that Conv2D will accept it
print('train_imgs.shape', train_imgs.shape)
test_imgs = lefts_and_rights[:test_split_index,:-1]
test_imgs = np.expand_dims(test_imgs, axis=3)

train_labels = labels[test_split_index:]
test_labels = labels[:test_split_index]

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(INPUT_HEIGHT, INPUT_WIDTH, 1)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    #kernel_regularizer=tf.keras.regularizers.l2(0.0001)
    #tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(2),
    tf.keras.layers.Activation(tf.keras.activations.sigmoid)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.MeanAbsoluteError(),
              metrics=['mean_absolute_error'])

model.fit(train_imgs, train_labels, epochs=2)

test_loss, test_acc = model.evaluate(test_imgs, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

print(model.predict(test_imgs[:10]),test_labels[:10])

# mae = tf.keras.losses.MeanAbsoluteError(tf.keras.losses.Reduction.SUM)
# mae([[0.,0.]],[[5.,1.]]) == 3.0
# Elementwise subtraction followed by sum
