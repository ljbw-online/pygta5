import numpy as np
import cv2
from time import time, sleep
import tensorflow as tf

rng = np.random.default_rng().integers

def imshow(ims, width, height, frame_rate):
    for im in ims:
        st = time()
        cv2.imshow('noise and lines',cv2.resize(im,(width,height),interpolation=cv2.INTER_NEAREST))
        if cv2.waitKey(25) == ord('q'):
            cv2.destroyAllWindows()
            break
        sleep(max(0, 1/frame_rate - (time() - st)))
    cv2.destroyAllWindows()

def make_data():
    length = 200000
    imgs = rng(0,256,length*32*32,dtype='uint8').reshape((length,32,32))
    # imgs = np.zeros((length,32,32), dtype='uint8')

    for i in range(0,length,2):
        imgs[i] = cv2.line(imgs[i], (rng(0,32),0), (rng(0,32),31), 255)

    labels = np.zeros(length,dtype='float32')
    labels[::2] = 1

    return imgs, labels

if __name__ == '__main__':
    train_images_255, train_labels = make_data()
    val_images, val_labels = make_data()
    test_images, test_labels = make_data()

    train_images = train_images_255.astype('float32') / 255.0
    val_images = val_images.astype('float32') / 255.0
    test_images = test_images.astype('float32') / 255.0

    imshow(train_images_255,320,320,2)

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(32, 32)),
        tf.keras.layers.Dense(32*32, activation='relu'),
        tf.keras.layers.Dense(1),
        tf.keras.layers.Activation(tf.keras.activations.sigmoid)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=4)

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print('\nTest accuracy:', test_acc)

    first_ten_predictions = model.predict(test_images[:10])
    print(first_ten_predictions)

    for i in range(10):
        cv2.imshow(
            'noise and lines {}'.format(i),
            cv2.resize(
                (test_images[i] * 255.0).astype('uint8'),
                (320, 320),
                interpolation=cv2.INTER_NEAREST)
        )
        if cv2.waitKey(0) == ord('q'):
            cv2.destroyAllWindows()

    # lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
    #     0.001,
    #     decay_steps=1000,
    #     decay_rate=1,
    #     staircase=False)
    #
    # def get_optimizer():
    #     return tf.keras.optimizers.Adam(lr_schedule)
