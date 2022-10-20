import tensorflow.keras as ks
from tensorflow.keras.models import load_model
import numpy as np

from common import MODEL_NAME

model = load_model(MODEL_NAME)

train_imgs = np.load('training_images.npy')
train_labels = np.load('training_labels.npy')
test_imgs = np.load('test_images.npy')
test_labels = np.load('test_labels.npy')

if __name__ == '__main__':
    metric_name = 'accuracy'
    model.compile(optimizer='adam',
                  loss=ks.losses.CategoricalCrossentropy(),
                  metrics=[metric_name])

    model.fit(train_imgs, train_labels, epochs=3)

    for prediction, label in zip(model.predict(test_imgs[:15]), test_labels[:15]):
        print(list(prediction), label)

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

# mae = tf.keras.losses.MeanAbsoluteError(tf.keras.losses.Reduction.SUM)
# mae([[0.,0.]],[[5.,1.]]) == 3.0
# Elementwise subtraction followed by sum