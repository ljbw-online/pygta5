from pathlib import Path
import os
from time import sleep, time

import numpy as np
import cv2
import tensorflow
import tensorflow.keras as ks

import windows

tensorflow.compat.v1.enable_eager_execution()

downloads = str(Path.home()) + '\\Downloads\\'

cap = cv2.VideoCapture(downloads + 'bouncing dvd logo-B5Jm716kuFc.mp4')
frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

frameShape = (frameHeight, frameWidth, 3)
videoShape = (frameCount,) + frameShape

downscale_width = 160
downscale_height = 90
downscale_shape = (downscale_height, downscale_width, 3)

inputsShape = (frameCount - 4, 4) + downscale_shape
labelsShape = (frameCount - 4,) + downscale_shape

print(inputsShape, labelsShape)

if os.path.isfile('training_inputs.npy'):
    cap.release()
    print('inputsShape', inputsShape)
    print('labelsShape', labelsShape)
    training_inputs = np.memmap('training_inputs.npy', mode='r', dtype=np.float32, shape=inputsShape)
    training_labels = np.memmap('training_labels.npy', mode='r', dtype=np.float32, shape=labelsShape)
else:
    buffer = np.empty((frameCount, downscale_height, downscale_width, 3), dtype=np.float32)

    fc = 0
    # return_val is false if there aren't any frames left, but cv2 has already told us exactly how many frames there are
    # return_val = True

    for i, frame in enumerate(buffer):
        return_val, video_frame = cap.read()

        buffer[i] = cv2.resize(video_frame, (downscale_width, downscale_height),
                               interpolation=cv2.INTER_NEAREST).astype(np.float32) / 255

        # print(buffer[i].dtype, buffer[i].shape, np.max(buffer[i]))

    cap.release()

    training_inputs = np.memmap('training_inputs.npy', dtype=np.float32, mode='w+',
                                shape=(frameCount - 4, 4, downscale_height, downscale_width, 3))

    training_labels = np.memmap('training_labels.npy', dtype=np.float32, mode='w+',
                                shape=(frameCount - 4, downscale_height, downscale_width, 3))
    for i, frame in enumerate(buffer[4:]):
        j = i + 4
        training_inputs[i] = buffer[(j - 4):j]
        training_labels[i] = buffer[j]

print(training_inputs.shape, training_labels.shape)


def imshow(ims, width=1280, height=None, frame_rate=0, title='imshow', sequences=False):
    if height is None:
        height = ims[0].shape[0] * round(width / ims[0].shape[1])

    for im in ims:
        t = time()
        im = cv2.resize(im, (width, height), interpolation=cv2.INTER_NEAREST)
        cv2.imshow(title, im)

        if frame_rate == 0:
            if cv2.waitKey(0) == ord('q'):
                cv2.destroyAllWindows()
                return
        else:
            if cv2.waitKey(25) == ord('q'):
                cv2.destroyAllWindows()
                return
            sleep(max(0., 1/frame_rate - (time() - t)))

    if not sequences:
        cv2.destroyAllWindows()


def imshow_seq(seq_array, width=1280, frame_rate=4):
    for seq in seq_array:
        imshow(seq, width=width, frame_rate=frame_rate, title='inputs', sequences=True)
        if cv2.waitKey(0) == ord('q'):
            cv2.destroyAllWindows()
            return


# imshow(training_labels, frame_rate=30, title='labels')
# imshow_seq(training_inputs, frame_rate=4)

model_input = windows.Input(shape=(4, downscale_height, downscale_width, 3))
x = ks.layers.ConvLSTM2D(3, (5, 5), padding='same', data_format='channels_last')(model_input)
x = ks.layers.Activation('tanh')(x)

model_output = x

model = ks.Model(inputs=[model_input], outputs=[model_output])
print(model.summary())
model.compile(loss=ks.losses.BinaryCrossentropy())
batch_size = 512

model.fit(training_inputs, training_labels, epochs=2, shuffle=True, batch_size=batch_size)

print('Evaluating:')
print(model.evaluate(training_inputs, training_labels, batch_size=batch_size, verbose=0))

eval_num = 256
predicted_frames = model.predict(training_inputs[:eval_num])
print(np.max(predicted_frames))
print(np.min(predicted_frames))

print(predicted_frames.dtype)

side_by_side = (np.concatenate((training_inputs[:eval_num, 3], predicted_frames), axis=2))
# side_by_side = (np.concatenate((training_inputs[:eval_num, 3], predicted_frames), axis=2) * 255).astype(np.uint8)

imshow(side_by_side)

# imshow((predicted_frames * 255).astype(np.uint8))

# print(side_by_side.shape)

video_writer = cv2.VideoWriter('video.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 30, (320, 90))

for frame in side_by_side:
    video_writer.write(frame)

video_writer.release()
