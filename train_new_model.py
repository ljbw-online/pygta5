import numpy as np
import tensorflow.keras as ks

from multihot_3 import get_model_definition, MODEL_NAME

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

model = get_model_definition()

training_images = np.load('training_images.npy')
training_labels = np.load('training_labels.npy')
test_images = np.load('test_images.npy')
test_labels = np.load('test_labels.npy')

print(model.summary())

print(training_images.shape, test_images.shape, training_labels.shape, test_labels.shape)
print(training_labels.dtype)

BATCH_SIZE = 64

if __name__ == '__main__':
    metric_name = 'accuracy'

    # learning_rate_schedule = ks.optimizers.schedules.ExponentialDecay(
    #     initial_learning_rate=0.5e-2,
    #     decay_steps=10000,
    #     decay_rate=0.9)
    # optimizer = keras.optimizers.SGD(learning_rate=lr_schedule)
    # optimizer=ks.optimizers.Adam(learning_rate=learning_rate_schedule),

    model.compile(loss=ks.losses.MeanSquaredError(), metrics=[metric_name])

    model.fit({'image_input': training_images}, {'model_output': training_labels}, epochs=1, batch_size=BATCH_SIZE)

    for prediction, label in zip(model.predict({'image_input': test_images[:15]}), test_labels[:15]):
        print(prediction, label)

    while True:
        test_loss, test_metric = \
            model.evaluate({'image_input': test_images}, {'model_output': test_labels}, verbose=2)

        print('Test accuracy:', test_metric)

        continue_choice = input('Enter a number N to continue training for N epochs\n'
                                'Enter s to save and quit\n'
                                'Enter q to quit without saving\n'
                                '(N/s/q)\n')

        if continue_choice.isnumeric():
            model.fit({'image_input': training_images},
                      {'model_output': training_labels},
                      epochs=int(continue_choice),
                      batch_size=BATCH_SIZE)
        elif continue_choice == 's':
            model.save(MODEL_NAME)
            exit()
        elif continue_choice == 'q':
            exit()

if True:
    pass
# conv2d_params == (filters_in_previous_layer * kernel_width**2 + 1) * output_filters

# mae = tf.keras.losses.MeanAbsoluteError(tf.keras.losses.Reduction.SUM)
# mae([[0.,0.]],[[5.,1.]]) == 3.0
# Elementwise subtraction followed by sum
