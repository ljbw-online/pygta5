import os
from collections import deque
from socket import socket
from time import sleep, time
from pathlib import Path
import pickle
import multiprocessing

import numpy as np
import cv2

from common import get_keys, get_gta_window, w_bytes, wa_bytes, wd_bytes, s_bytes, sa_bytes, a_bytes, sd_bytes, \
d_bytes, nk_bytes, supervised_resumed_bytes, model_paused_bytes, uint8_to_float32

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
from tensorflow import keras as ks

input_shape = (4, 90, 160)


def create_model():
    inputs = ks.layers.Input(shape=input_shape)
    rescaled = ks.layers.Rescaling(1./255)(inputs)

    conv1_reg = 1 / (32 * 8 * 21 * 39)
    conv2_reg = 1 / (32 * 16 * 9 * 18)
    conv3_reg = 1 / (32 * 32 * 7 * 16)

    # Based on the Deep Q implementation for Breakout, on keras.io
    layer1 = ks.layers.Conv2D(8, 8, strides=4, activation='relu', data_format='channels_first',
                              activity_regularizer=ks.regularizers.l1(conv1_reg))(rescaled)

    layer2 = ks.layers.Conv2D(16, 4, strides=2, activation='relu', data_format='channels_first',
                              activity_regularizer=ks.regularizers.l1(conv2_reg))(layer1)

    layer3 = ks.layers.Conv2D(32, 3, strides=1, activation='relu', data_format='channels_first',
                              activity_regularizer=ks.regularizers.l1(conv3_reg))(layer2)

    layer4 = ks.layers.Flatten()(layer3)

    layer5 = ks.layers.Dense(32, activation='relu')(layer4)
    action_probs = ks.layers.Dense(9, activation=ks.activations.softmax)(layer5)

    return ks.Model(inputs=inputs, outputs=[action_probs, layer1])


def get_frame():
    frame = get_gta_window()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2GRAY)
    frame = cv2.resize(frame, (input_shape[2], input_shape[1]), interpolation=cv2.INTER_NEAREST)
    return frame


def apply_correction(keys, action):
    corrected = False
    # w wa wd s sa sd a d <no-keys>
    if 'I' in keys:
        corrected = True
        if 'J' in keys:
            action = 1
        elif 'L' in keys:
            action = 2
        else:
            action = 0
    elif 'K' in keys:
        corrected = True
        if 'J' in keys:
            action = 4
        elif 'L' in keys:
            action = 5
        else:
            action = 3
    elif 'J' in keys:
        corrected = True
        action = 6
    elif 'L' in keys:
        corrected = True
        action = 7
    # else: # Currently indistinguishable from me stopping my corrections to let the model drive.
    #     action = 8

    return action, corrected


def send_action(sock, action):
    if action == 0:
        sock.sendall(w_bytes)
    elif action == 1:
        sock.sendall(wa_bytes)
    elif action == 2:
        sock.sendall(wd_bytes)
    elif action == 3:
        sock.sendall(s_bytes)
    elif action == 4:
        sock.sendall(sa_bytes)
    elif action == 5:
        sock.sendall(sd_bytes)
    elif action == 6:
        sock.sendall(a_bytes)
    elif action == 7:
        sock.sendall(d_bytes)
    elif action == 8:
        sock.sendall(nk_bytes)


def action_to_bools(action):
    (w, a, s, d) = (False, False, False, False)
    match action:
        case 0:
            w = True
        case 1:
            w = True
            a = True
        case 2:
            w = True
            d = True
        case 3:
            s = True
        case 4:
            s = True
            a = True
        case 5:
            s = True
            d = True
        case 6:
            a = True
        case 7:
            d = True

    return [w, a, s, d]


optimizer = ks.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)  # clipnorm from Breakout example
loss_function = ks.losses.SparseCategoricalCrossentropy()


def update_model(inq, outq):
    batch_count = 1
    model = create_model()
    while True:
        q_item = inq.get(block=True)

        if q_item is None:
            # If a process has been putting stuff on a queue then Process.join() will hang until everything has been
            # removed from that queue. This method could be called allow_exit_without_flush().
            outq.cancel_join_thread()
            return
        else:
            (frame_stack_sample, correction_sample) = q_item

        with tf.GradientTape() as tape:
            predicted_action_sample, _ = model(frame_stack_sample)

            loss = loss_function(correction_sample, predicted_action_sample) + sum(model.losses)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if batch_count % 200 == 0:
            print('batch_count {}, loss: {:.2f}'.format(batch_count, loss))

        batch_count += 1

        outq.put(model.get_weights())


def visualise(q):
    last_corrected_time = time()
    wasd_im = np.zeros((360, 640), dtype=np.uint8)
    frames_convs = np.zeros((640, 1080), dtype=np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    pressed_colour = (170,)
    non_pressed_colour = (85,)
    letter_colour = (255,)

    square_top_lefts = [(220, 10), (10, 190), (220, 190), (430, 190)]
    square_bottom_rights = [(410, 170), (200, 350), (410, 350), (620, 350)]
    letter_bottom_lefts = [(243, 153), (33, 333), (243, 333), (453, 333)]
    letters = ['W', 'A', 'S', 'D']

    while True:
        q_item = q.get(block=True)

        if q_item is None:
            cv2.destroyAllWindows()
            return
        else:
            frame_list, conv_stack, action, corrected = q_item

        t = time()
        if corrected:
            last_corrected_time = time()

        recent_corrections = t - last_corrected_time < 1.0

        if recent_corrections:
            cv2.putText(frames_convs, 'Streamer', (25, 120), font, 3, (255,), 3)
            cv2.putText(frames_convs, 'is', (50, 190), font, 3, (255,), 3)
            cv2.putText(frames_convs, 'driving', (25, 260), font, 3, (255,), 3)
        else:
            # Only update frames & convs when I'm not correcting
            frame01 = np.hstack((frame_list[0], frame_list[1]))
            frame23 = np.hstack((frame_list[2], frame_list[3]))

            frames = np.vstack((frame01, frame23))
            frames = cv2.resize(frames, (640, 360), interpolation=cv2.INTER_NEAREST)
            frames = uint8_to_float32(frames)

            conv_list = []
            for i in range(0, len(conv_stack), 2):
                conv_list.append(np.hstack((conv_stack[i], conv_stack[i + 1])))

            convs = np.vstack(conv_list)
            convs = cv2.resize(convs, (640, 720), interpolation=cv2.INTER_NEAREST)

            frames_convs = np.vstack((frames, convs))

        cv2.imshow('INPUT & CONV1', frames_convs)

        wasd_bools = action_to_bools(action)
        wasd_zip = zip(square_top_lefts, square_bottom_rights, letters, wasd_bools, letter_bottom_lefts)

        for square_top_left, square_bottom_right, letter, letter_bool, letter_bottom_left in wasd_zip:
            if letter_bool:
                square_colour = pressed_colour
            else:
                square_colour = non_pressed_colour

            cv2.rectangle(wasd_im, square_top_left, square_bottom_right, square_colour, cv2.FILLED)
            cv2.putText(wasd_im, letter, letter_bottom_left, font, 6, letter_colour, 8)

        if recent_corrections:
            streamer_text_colour = (255,)
            ai_text_colour = (85,)
        else:
            streamer_text_colour = (85,)
            ai_text_colour = (255,)

        cv2.putText(wasd_im, 'Streamer', (25, 100), font, 1, streamer_text_colour, 2)
        cv2.putText(wasd_im, 'AI', (505, 100), font, 1, ai_text_colour, 2)

        cv2.imshow('WASD', wasd_im)
        cv2.waitKey(1)


def main(resume=False):
    file_name = os.path.basename(__file__.removesuffix('.py'))
    model_path = os.path.join(Path.home(), 'My Drive\\Models',  file_name)
    data_path = os.path.join(Path.home(), 'Documents\\Data', file_name)

    frame_stacks = []
    action_probs_history = []
    correction_history = []

    if resume:
        model = ks.models.load_model(model_path)

        with open(data_path, 'rb') as file:
            data = pickle.load(file)

        frame_stacks = data['frame_stacks']
        correction_history = data['correction_history']
    else:
        model = create_model()

    model.summary()

    batch_size = 32
    max_history_len = 100_000

    frame_deque = deque(maxlen=input_shape[0])

    for _ in range(4):
        frame_deque.append(get_frame())

    batch_q = multiprocessing.Queue()
    weights_q = multiprocessing.Queue()

    training_process = multiprocessing.Process(target=update_model, args=(batch_q, weights_q))
    training_process.start()
    weights_q.put(model.get_weights())

    image_q = multiprocessing.Queue()
    visualisation_process = multiprocessing.Process(target=visualise, args=(image_q,))
    visualisation_process.start()

    sock = socket()
    while True:
        try:
            sock.connect(("127.0.0.1", 7001))
            break
        except ConnectionRefusedError:
            print("Connection Refused")
            sleep(1)

    sock.sendall(supervised_resumed_bytes)

    frame_count = 0

    num_corrections = 0
    num_corrections_saved = 0
    training = True
    last_corrected_time = time()
    get_keys()
    while True:
        keys = get_keys()
        frame_count += 1

        frame_deque.append(get_frame())
        frame_list = list(frame_deque)

        state = tf.convert_to_tensor(frame_list)
        state = tf.expand_dims(state, 0)
        action_probs, conv2d = model(state, training=False)
        action_probs = action_probs[0]
        conv2d = conv2d[0]

        model_action = tf.argmax(action_probs)

        action, corrected = apply_correction(keys, model_action)
        send_action(sock, action)

        if corrected:
            last_corrected_time = time()
            num_corrections += 1
            if model_action != action:
                num_corrections_saved += 1
                frame_stacks.append(frame_list)
                correction_history.append(action)
            # action_probs_history.append(action_probs)

        if training and not weights_q.empty() and len(correction_history) > batch_size:
            model.set_weights(weights_q.get())

            indices = np.random.choice(range(len(correction_history)), size=batch_size)

            frame_stack_sample = np.array([frame_stacks[i] for i in indices])
            correction_sample = np.array([correction_history[i] for i in indices])
            # action_probs_sample = [action_probs_history[i] for i in indices]

            batch_q.put((frame_stack_sample, correction_sample))

        if image_q.empty():
            image_q.put((frame_list, conv2d, action, corrected))

        if len(action_probs_history) > max_history_len:
            del frame_stacks[:1]
            del action_probs_history[:1]

        if num_corrections == 100:
            print('{}% of corrections saved'.format(int(100 * num_corrections_saved / num_corrections)))
            num_corrections = 0
            num_corrections_saved = 0

        if '1' in keys:
            user_paused = True
            training = False
            print('Stopped training. {} frame stacks'.format(len(frame_stacks)))
        elif '2' in keys:
            user_paused = False

        t = time()
        if (t - last_corrected_time < 1) or (t - last_corrected_time > 180):
            training = False
        else:
            training = True

        if '5' in keys:
            sock.sendall(model_paused_bytes)
            choice = input('Quit/Save-and-close/Continue (q/s/C)')
            if choice == 'q':
                sock.close()
                batch_q.put(None)
                training_process.join()
                image_q.put(None)
                visualisation_process.join()
                break
            elif choice == 's':
                sock.close()
                batch_q.put(None)
                training_process.join()
                image_q.put(None)
                visualisation_process.join()

                model.save(model_path)
                with open(data_path, 'wb') as file:
                    pickle.dump({'frame_stacks': frame_stacks, 'correction_history': correction_history}, file)
                break
            else:
                sock.sendall(supervised_resumed_bytes)


if __name__ == '__main__':
    main()
