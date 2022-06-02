import numpy as np
import cv2
from time import time, sleep
from itertools import repeat

rng = np.random.default_rng().integers


def imshow(ims, width, height=None, frame_rate=0, labels=None, title='noise and lines'):
    if height is None:
        height = width

    if labels is None:
        labels = repeat(None, len(ims))

    for im, label in zip(ims, labels):
        st = time()

        if label is not None:
            print(label)

        cv2.imshow(title, cv2.resize(im, (width, height), interpolation=cv2.INTER_NEAREST))
        if frame_rate == 0:
            if cv2.waitKey(0) == ord('q'):
                cv2.destroyAllWindows()
                break
        else:
            if cv2.waitKey(25) == ord('q'):
                cv2.destroyAllWindows()
                break
            sleep(max(0, round(1/frame_rate - (time() - st))))

    cv2.destroyAllWindows()


def make_data(img_size, non_lines_prop=1):
    lines_num = img_size ** 2 * 2
    non_lines_num = lines_num * non_lines_prop

    noise_amplitude = 256  # 256 for max amplitude

    non_lines_imgs = rng(
        0, noise_amplitude, non_lines_num * img_size * img_size, dtype='uint8').reshape((lines_num, img_size, img_size))

    lines_imgs = rng(
        0, noise_amplitude, lines_num * img_size * img_size, dtype='uint8').reshape((lines_num, img_size, img_size))

    # Top to bottom
    for i in range(0, img_size):
        for j in range(0, img_size):
            lines_imgs[i * img_size + j] = cv2.line(lines_imgs[i * img_size + j], (i, 0), (j, img_size - 1), 255)

    # Left to right
    for i in range(0, img_size):
        for j in range(0, img_size):
            lines_imgs[(img_size ** 2) + i * img_size + j] = cv2.line(
                lines_imgs[(img_size ** 2) + i * img_size + j], (0, i), (img_size - 1, j), 255)

    lbls = np.concatenate((np.ones(lines_num, dtype='float32'), np.zeros(non_lines_num, dtype='float32')))

    return np.concatenate((lines_imgs, non_lines_imgs)), lbls


IMG_SIZE = 32


def main():
    images, labels = make_data(IMG_SIZE)

    # print(labels[:10])
    # imshow(images[:(IMG_SIZE ** 2 * 2)], 500)
    #
    # print(labels[(IMG_SIZE ** 2 * 2 - 1):(IMG_SIZE ** 2 * 2 + 10)])
    # imshow(images[(IMG_SIZE ** 2 * 2 - 1):], 500, title='noise without lines')

    images = images.astype('float32') / 255  # RESCALE

    images = np.expand_dims(images, axis=3)

    import tensorflow as tf
    import tensorflow.keras as ks
    tf.enable_eager_execution()
    print(images.dtype, images.shape)

    model_input = ks.Input(shape=(IMG_SIZE, IMG_SIZE, 1))
    # x = ks.layers.Rescaling(scale=1./255)(model_input)
    x = ks.layers.Conv2D(1, (5, 5), padding='same', activation='relu', kernel_regularizer=ks.regularizers.l2(0.0001))(model_input)
    x = ks.layers.MaxPooling2D()(x)
    x = ks.layers.Conv2D(1, (5, 5), padding='same', activation='relu', kernel_regularizer=ks.regularizers.l2(0.0001))(x)
    x = ks.layers.MaxPooling2D()(x)
    x = ks.layers.Flatten()(x)
    x = ks.layers.Dense(16, activation='relu', kernel_regularizer=ks.regularizers.l2(0.0001))(x)
    # x = ks.layers.Dense(128, activation='relu')(x)
    model_output = ks.layers.Dense(1, activation='sigmoid')(x)

    model = ks.Model(inputs=[model_input], outputs=[model_output])

    print(model.summary())

    # train_metrics = [ks.metrics.Recall()]
    train_metrics = [ks.metrics.Recall()]
    # opt = ks.optimizers.Adam(learning_rate=0.1)
    model.compile(loss=ks.losses.MeanSquaredError(), metrics=train_metrics)#, optimizer=opt)

    model.fit(images, labels, epochs=100, shuffle=True, batch_size=64)

    print('Evaluating:')
    model.evaluate(images, labels)

    model_labels = model.predict(images)
    # print(model_labels[:20])
    # exit()

    # true_false_metrics = [ks.metrics.TruePositives(), ks.metrics.FalsePositives(),
    #                       ks.metrics.TrueNegatives(), ks.metrics.FalseNegatives()]

    model = ks.Model(inputs=model.inputs, outputs=[layer.output for layer in model.layers])

    eval_num = 50

    lines_conv_output = model(images[:eval_num])[1].numpy()
    lines_label_output = model(images[:eval_num])[-1].numpy()
    # lines_float32 = images[:eval_num].astype('float32') / 255
    lines_float32 = images[:eval_num]
    print(lines_float32.shape, lines_conv_output.shape)
    side_by_side = np.concatenate((lines_float32, lines_conv_output), axis=2)
    imshow(side_by_side, 1000, height=500, labels=lines_label_output)

    start_point = IMG_SIZE ** 2 * 2
    non_lines_conv_output = model(images[start_point:(start_point + eval_num)])[1].numpy()
    non_lines_label_output = model(images[start_point:(start_point + eval_num)])[-1].numpy()
    # non_lines_float32 = images[start_point:(start_point + eval_num)].astype('float32') / 255
    non_lines_float32 = images[start_point:(start_point + eval_num)]
    side_by_side = np.concatenate((non_lines_float32, non_lines_conv_output), axis=2)
    imshow(side_by_side, 1000, height=500, labels=non_lines_label_output)

    unsure_num = 0
    unsure_ims = np.zeros_like(images)
    unsure_convs = np.zeros_like(images).astype('float32')
    unsure_lbls = np.zeros_like(labels)

    for i in range(len(model_labels)):
        if (model_labels[i] < 0.6) and (model_labels[i] > 0.4):
            unsure_ims[unsure_num] = images[i]
            unsure_convs[unsure_num] = model(np.expand_dims(images[i], axis=0))[1].numpy()
            unsure_lbls[unsure_num] = model(np.expand_dims(images[i], axis=0))[-1].numpy()

            unsure_num += 1

    if unsure_num == 0:
        print('model unsure about no images')
    else:
        print(unsure_num)
        # unsure_ims = unsure_ims.astype('float32') / 255  # Images already rescaled for tf v1
        unsure_ims = unsure_ims[:unsure_num]
        unsure_convs = unsure_convs[:unsure_num]
        unsure_lbls = unsure_lbls[:unsure_num]

        side_by_side = np.concatenate((unsure_ims, unsure_convs), axis=2)
        imshow(side_by_side, 1000, height=500, labels=unsure_lbls, title='UNSURE')

    sure_num = 0
    sure_ims = np.zeros_like(images)
    sure_convs = np.zeros_like(images).astype('float32')
    sure_lbls = np.zeros_like(labels)

    for i in range(len(model_labels)):
        if (model_labels[i] < 0.1) or (model_labels[i] > 0.9):
            sure_ims[sure_num] = images[i]
            sure_convs[sure_num] = model(np.expand_dims(images[i], axis=0))[1].numpy()
            sure_lbls[sure_num] = model(np.expand_dims(images[i], axis=0))[-1].numpy()

            sure_num += 1
    # REMEMBER MODEL NOT NECESSARILY CORRECT ABOUT CLASSIFICATION
    if sure_num == 0:
        print('model not sure about any images')
    else:
        print(sure_num)
        # sure_ims = sure_ims.astype('float32') / 255  # Images already rescaled for tf v1
        sure_ims = sure_ims[:sure_num]
        sure_convs = sure_convs[:sure_num]
        sure_lbls = sure_lbls[:sure_num]
        print(sure_ims.shape, sure_convs.shape)

        side_by_side = np.concatenate((sure_ims, sure_convs), axis=2)
        imshow(side_by_side, 1000, height=500, labels=sure_lbls, title='SURE')


if __name__ == '__main__':
    main()
