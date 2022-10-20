import os

import numpy as np
import cv2
import tensorflow as tf
import tensorflow.keras as ks
from PIL import Image

from common import imshow

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
tf.compat.v1.enable_eager_execution()

rng = np.random.default_rng().integers
FONT = cv2.FONT_HERSHEY_SIMPLEX


def make_data(img_size, non_lines_prop=1, noise_amplitude=256):
    lines_num = img_size ** 2 * 2
    non_lines_num = round(lines_num * non_lines_prop)

    noise_amplitude = noise_amplitude  # 256 for max amplitude
    if noise_amplitude == 0:
        noise_amplitude = 1

    if non_lines_prop > 0:
        non_lines_imgs = rng(0, noise_amplitude, non_lines_num * img_size * img_size * 3, dtype='uint8')
        non_lines_imgs = non_lines_imgs.reshape((lines_num, img_size, img_size, 3))
    else:
        non_lines_imgs = np.zeros((0, img_size, img_size, 3), dtype=np.uint8)

    lines_imgs = rng(0, noise_amplitude, lines_num * img_size * img_size * 3, dtype='uint8')
    lines_imgs = lines_imgs.reshape((lines_num, img_size, img_size, 3))

    lbls = np.zeros((lines_num + non_lines_num, img_size, img_size, 3), dtype='uint8')

    # Top to bottom
    for i in range(0, img_size):
        for j in range(0, img_size):
            lines_imgs[i * img_size + j] = cv2.line(lines_imgs[i * img_size + j], (i, 0), (j, img_size - 1),
                                                    (255, 255, 255), 2)

            lbls[i * img_size + j] = cv2.line(lbls[i * img_size + j], (i, 0), (j, img_size - 1),
                                              (255, 255, 255), 2)

    # Left to right
    for i in range(0, img_size):
        for j in range(0, img_size):
            lines_imgs[(img_size ** 2) + i * img_size + j] = cv2.line(
                lines_imgs[(img_size ** 2) + i * img_size + j], (0, i), (img_size - 1, j), (255, 255, 255), 2)

            lbls[(img_size ** 2) + i * img_size + j] = cv2.line(
                lbls[(img_size ** 2) + i * img_size + j], (0, i), (img_size - 1, j), (255, 255, 255), 2)

    ims = np.concatenate((lines_imgs, non_lines_imgs))
    # lbls = np.concatenate((np.ones(lines_num, dtype='float32'), np.zeros(non_lines_num, dtype='float32')))

    data = np.zeros(lines_num + non_lines_num,
                    dtype=[('image', np.float32, (img_size, img_size, 3)),
                           ('label', np.float32, (img_size, img_size, 3))])
    data = np.rec.array(data)

    data.image = ims.astype(np.float32) / 255
    data.label = lbls.astype(np.float32) / 255

    return data


# putText doesn't really work on very small rects, gonna have to overload common.imshow :( :( :(
def rect_with_text(width, height, text):
    rect = np.zeros((height, width, 3), dtype=np.float32)
    return cv2.putText(rect, text, (round(height * 0.3), round(width * 0.3)), FONT, 1, (1.0, 1.0, 1.0), 1)


def custom_loss(a, b):
    # ABS DIFFERENCE + 1/TF.REDUCE_SUM(B)
    b_sums = tf.reduce_sum(b, axis=(1, 2, 3)) + 0.0001
    nums = tf.ones_like(b_sums) * 512

    nums_over_b_sums = tf.divide(nums, b_sums)

    abs_diffs = tf.reduce_sum(tf.abs(a - b), axis=(1, 2, 3))
    return abs_diffs + nums_over_b_sums


IMG_SIZE = 64


def main():
    data = make_data(IMG_SIZE, non_lines_prop=0, noise_amplitude=32)

    print(data.image.dtype, data.image.shape, data.label.dtype, data.label.shape)

    l2 = ks.regularizers.l2

    model_input = ks.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    # x = ks.layers.Conv2D(3, (5, 5), padding='same', activation='relu',
    #                      kernel_regularizer=l2(0.0001), data_format='channels_last')(model_input)
    # x = ks.layers.MaxPooling2D(data_format='channels_last')(x)
    # x = ks.layers.Conv2D(3, (5, 5), padding='same', activation='relu',
    #                      kernel_regularizer=l2(0.0001), data_format='channels_last')(x)
    # x = ks.layers.MaxPooling2D(data_format='channels_last')(x)
    # x = ks.layers.Flatten()(x)
    # x = ks.layers.Dense(16, activation='relu', kernel_regularizer=l2(0.0001))(x)
    # model_output = ks.layers.Dense(1, activation='sigmoid')(x)
    # model_output = x

    # x = ks.layers.Flatten()(model_input)
    # x = ks.layers.Dense(IMG_SIZE * IMG_SIZE * 3 * 1, activation='relu', kernel_regularizer=l2(0.0001))(x)
    # x = ks.layers.Dense(IMG_SIZE * IMG_SIZE * 3, activation='sigmoid', kernel_regularizer=l2(0.0001))(x)
    # x = ks.layers.Reshape((IMG_SIZE, IMG_SIZE, 3))(x)
    # model_output = x

    x = ks.layers.Conv2D(3, (5, 5), padding='same', activation='relu',
                         data_format='channels_last', kernel_regularizer=l2(0.0001))(model_input)
    x = ks.layers.Conv2D(1, (5, 5), padding='same', activation='relu',
                         data_format='channels_last', kernel_regularizer=l2(0.0001))(x)
    x = ks.layers.Flatten()(x)
    x = ks.layers.Dense(IMG_SIZE * IMG_SIZE * 3, activation='sigmoid', kernel_regularizer=l2(0.0001))(x)
    x = ks.layers.Reshape((IMG_SIZE, IMG_SIZE, 3))(x)
    model_output = x

    model = ks.Model(inputs=[model_input], outputs=[model_output])

    print(model.summary())

    # train_metrics = [ks.metrics.Recall()]
    # model.compile(loss=ks.losses.MeanSquaredError())  # , metrics=train_metrics)  # , optimizer=opt)
    model.compile(loss=ks.losses.BinaryCrossentropy())

    model.fit(data.image, data.label, epochs=8, shuffle=True, batch_size=512)

    print('Evaluating:')
    print(model.evaluate(data.image, data.label, batch_size=8192, verbose=0))

    # model_labels = model.predict(data.image)

    model = ks.Model(inputs=model.inputs, outputs=[layer.output for layer in model.layers])

    eval_num = 128

    inputs_float32 = data.image[:eval_num]
    labels_float32 = data.label[:eval_num]

    model_outputs = model(inputs_float32)
    conv1_float32 = model_outputs[1].numpy()
    conv2_float32 = model_outputs[2].numpy()
    conv2_float32 = np.repeat(conv2_float32, 3, axis=3)
    print(conv2_float32.shape)
    predictions_float32 = model_outputs[-1].numpy()

    # input_rect = rect_with_text(data.image[0].shape[1], round(data.image[0].shape[0] * 0.2), 'input')
    # label_rect = rect_with_text(data.label[0].shape[1], round(data.label[0].shape[0] * 0.2), 'label')
    # prediction_rect = rect_with_text(predictions_float32[0].shape[1], round(predictions_float32[0].shape[0] * 0.2),
    #                                  'prediction')
    #
    # input_rect = np.expand_dims(input_rect, axis=0)
    # label_rect = np.expand_dims(label_rect, axis=0)
    # prediction_rect = np.expand_dims(prediction_rect, axis=0)
    #
    # input_rects = np.repeat(input_rect, eval_num, axis=0)
    # label_rects = np.repeat(label_rect, eval_num, axis=0)
    # prediction_rects = np.repeat(prediction_rect, eval_num, axis=0)
    # print(inputs_float32.shape)
    # inputs_float32 = np.concatenate((inputs_float32, input_rects), axis=1)
    # labels_float32 = np.concatenate((labels_float32, label_rects), axis=1)
    # predictions_float32 = np.concatenate((predictions_float32, prediction_rects), axis=1)

    side_by_side = np.concatenate((conv1_float32, conv2_float32, predictions_float32), axis=2)
    number_of_imgs = round(side_by_side.shape[2] / IMG_SIZE)

    data_to_imshow = np.rec.array(np.zeros(eval_num,
                                  dtype=[('image', np.float32, side_by_side[0].shape), ('label', np.float32)]))

    data_to_imshow.image = side_by_side

    imshow(data_to_imshow, 1536)

    save_width = 1024

    choice = input('Save? (y/N)\n')

    if choice == 'y':
        data_to_save = np.zeros((eval_num, IMG_SIZE * round(save_width / (IMG_SIZE * number_of_imgs)), save_width, 3),
                                dtype=data_to_imshow.image.dtype)
        print(data_to_save.shape)

        for i, im in enumerate(data_to_imshow.image):
            data_to_save[i] = cv2.resize(im, (save_width, IMG_SIZE * round(save_width / (IMG_SIZE * number_of_imgs))),
                                         interpolation=cv2.INTER_NEAREST)

        imgs = [Image.fromarray(img) for img in (data_to_save * 255).astype(np.uint8)]
        imgs[0].save("noise_and_lines.gif", save_all=True, append_images=imgs[1:], duration=60, loop=0)


if __name__ == '__main__':
    main()
