from common import INPUT_WIDTH, INPUT_HEIGHT, MODEL_NAME, OUTPUT_LENGTH
import tensorflow.keras as ks
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# MODEL_NAME = ''

# model = ks.Sequential([
#     # ks.layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(INPUT_HEIGHT, INPUT_WIDTH, 1)),
#     # # kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
#     # # tf.keras.layers.Dropout(0.5),
#     # ks.layers.MaxPooling2D(),
#     ks.layers.Conv2D(32, 3, padding='same', activation='relu', input_shape=(INPUT_HEIGHT, INPUT_WIDTH, 1)),
#     # kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
#     # tf.keras.layers.Dropout(0.5),
#     ks.layers.MaxPooling2D(),
#     ks.layers.Conv2D(64, 3, padding='same', activation='relu'),
#     # kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
#     # tf.keras.layers.Dropout(0.5),
#     ks.layers.MaxPooling2D(),
#     ks.layers.Flatten(),
#     ks.layers.Dense(16, activation='relu'),
#     # tf.keras.layers.Dropout(0.5),
#     ks.layers.Dense(4, activation='softmax'),
# ])

# conv2d_params == (filters_in_previous_layer * kernel_width**2 + 1) * output_filters

image_input = ks.Input(shape=(INPUT_HEIGHT, INPUT_WIDTH, 1), name='image_input')
x = ks.layers.Conv2D(32, 4, padding='same', activation='relu')(image_input)
x = ks.layers.MaxPooling2D()(x)
x = ks.layers.Conv2D(64, 4, padding='same', activation='relu')(x)
x = ks.layers.MaxPooling2D()(x)
x = ks.layers.Flatten()(x)
x = ks.layers.Dense(16, activation='relu')(x)

# keys_input = ks.Input(shape=(INPUT_WIDTH - OUTPUT_LENGTH,), name='keys_input')
# y = ks.layers.Dense(16, activation='relu')(keys_input)
# z = ks.layers.concatenate([x, y])

model_output = ks.layers.Dense(4, activation='softmax')(x)

model = ks.Model(inputs=image_input, outputs=model_output)

train_imgs = np.load('training_images.npy')
# train_keys = np.load('training_key_histories.npy')
train_labels = np.load('training_labels.npy')
test_imgs = np.load('test_images.npy')
# test_keys = np.load('test_key_histories.npy')
test_labels = np.load('test_labels.npy')

if __name__ == '__main__':
    metric_name = 'accuracy'
    model.compile(optimizer='adam',
                  loss=ks.losses.CategoricalCrossentropy(),
                  metrics=[metric_name])

    model.fit({'image_input': train_imgs}, train_labels, epochs=1)

    for prediction, label in zip(model.predict({'image_input': test_imgs[:15]}), test_labels[:15]):
        print(list(prediction), label)

    while True:
        test_loss, test_metric = model.evaluate({'image_input': test_imgs},
                                                test_labels,
                                                verbose=2)
        print('Test accuracy:', test_metric)

        continue_choice = input('Enter a number N to continue training for N epochs\n'
                                'Enter s to save and quit\n'
                                'Enter q to quit without saving\n'
                                '(N/s/q)\n')

        if continue_choice.isnumeric():
            model.fit({'image_input': train_imgs}, train_labels, epochs=int(continue_choice))
        elif continue_choice == 's':
            model.save(MODEL_NAME)
            exit()
        elif continue_choice == 'q':
            exit()
        else:
            print('huh?')

# mae = tf.keras.losses.MeanAbsoluteError(tf.keras.losses.Reduction.SUM)
# mae([[0.,0.]],[[5.,1.]]) == 3.0
# Elementwise subtraction followed by sum
