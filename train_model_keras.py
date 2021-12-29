from common import INPUT_WIDTH, INPUT_HEIGHT, MODEL_NAME
import tensorflow.keras as ks
import numpy as np

model = ks.Sequential([
    ks.layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(INPUT_HEIGHT, INPUT_WIDTH, 1)),
                           # kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
    # tf.keras.layers.Dropout(0.5),
    ks.layers.MaxPooling2D(),
    ks.layers.Conv2D(32, 3, padding='same', activation='relu'),
                           # kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
    # tf.keras.layers.Dropout(0.5),
    ks.layers.MaxPooling2D(),
    ks.layers.Conv2D(64, 3, padding='same', activation='relu'),
                           # kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
    # tf.keras.layers.Dropout(0.5),
    ks.layers.MaxPooling2D(),
    ks.layers.Flatten(),
    ks.layers.Dense(128, activation='relu'),
    # tf.keras.layers.Dropout(0.5),
    ks.layers.Dense(4, activation='softmax'),
])

train_imgs = np.load('train_imgs.npy')
train_labels = np.load('train_labels.npy')
test_imgs = np.load('test_imgs.npy')
test_labels = np.load('test_labels.npy')

if __name__ == '__main__':
    metric_name = 'accuracy'
    model.compile(optimizer='adam',
                  loss=ks.losses.CategoricalCrossentropy(),
                  metrics=[metric_name])

    model.fit(train_imgs, train_labels, epochs=3)

    test_loss, test_metric = model.evaluate(test_imgs, test_labels, verbose=2)
    print('\n{} on test set:'.format(metric_name), test_metric)

    for prediction, label in zip(model.predict(test_imgs[:15]), test_labels[:15]):
        print(list(prediction), label)

    model.save(MODEL_NAME)

# mae = tf.keras.losses.MeanAbsoluteError(tf.keras.losses.Reduction.SUM)
# mae([[0.,0.]],[[5.,1.]]) == 3.0
# Elementwise subtraction followed by sum
