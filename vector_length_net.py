import math

import numpy as np
import tensorflow as tf
import tensorflow.keras as ks

np.set_printoptions(precision=3, floatmode='fixed', suppress=True, sign=' ')
rng = np.random.default_rng()

l2 = ks.regularizers.l2(1.0)

model_input = ks.layers.Input((1,))
dense1 = ks.layers.Dense(16, activation='relu', kernel_regularizer=l2)(model_input)
dense2 = ks.layers.Dense(16, activation='relu', kernel_regularizer=l2)(dense1)
# x = ks.layers.Dense(32, activation='relu')(x)
model_output = ks.layers.Dense(1)(dense2)

model = ks.Model(inputs=model_input, outputs=model_output)
model.compile()
optimizer = ks.optimizers.Adam(learning_rate=0.1)

def vec_len(a):
    return np.sqrt(np.sum(a**2))

comp_max_abs = 1.0

def generate_batch(n):
    return (rng.random((n, 1), dtype=np.float32))  # * 2.0 * comp_max_abs) - comp_max_abs

episode_num = 0
lr_reduced = False
lr_reduced_again = False
batch_size = 1000

while True:
    vec_lens = []
    outputs = []
    error_list = []
    with tf.GradientTape() as tape:
    # if True:
    #     rnd_vecs = (rng.random((100, 3), dtype=np.float32))  # - 0.5) * 200
        for v in generate_batch(batch_size):
            v_len_tensor = tf.convert_to_tensor(vec_len(v), dtype=tf.float32)
            output = model(tf.expand_dims(v, 0), training=True)[0, 0]
            vec_lens.append(v_len_tensor)
            outputs.append(output)

            error_list.append(abs(v_len_tensor - output))

        errors = sum(error_list)

        loss = errors + sum(model.losses)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

    for v_len_tensor, output_tensor in zip(vec_lens[:5], outputs[:5]):
        print('actual: {: 4.3}, net: {: 4.3}'.format(v_len_tensor.numpy(), output_tensor.numpy()))

    episode_num += 1
    print('errors {: 5.4}'.format(errors.numpy()))
    print('loss {: 5.4}'.format(loss.numpy()))

    if errors.numpy() < (0.1 * batch_size) and not lr_reduced:
        print('Reducing learning rate to 0.01')
        optimizer = ks.optimizers.Adam(learning_rate=0.01)
        lr_reduced = True

    if errors.numpy() < (0.01 * batch_size) and not lr_reduced_again:
        print('Reducing learning rate to 0.001')
        optimizer = ks.optimizers.Adam(learning_rate=0.001)
        lr_reduced_again = True

    if errors.numpy() < 0.001 * batch_size:
        # for v_len_tensor, output_tensor in zip(vec_lens, outputs):
        #     print('actual: {: 4.3}, net: {: 4.3}, error: {: 4.3}'.format(
        #         v_len_tensor.numpy(), output_tensor.numpy(), abs(v_len_tensor.numpy() - output_tensor.numpy())))

        with tf.GradientTape() as tape:
            for v in generate_batch(1000):
                v_len_tensor = tf.convert_to_tensor(vec_len(v), dtype=tf.float32)
                output = model(tf.expand_dims(v, 0))[0, 0]
                vec_lens.append(v_len_tensor)
                outputs.append(output)

                error_list.append(abs(v_len_tensor - output))

            errors = sum(error_list)

            loss = errors + sum(model.losses)

        print('errors with 1000 vectors: {: 4.3}'.format(errors.numpy()))

        model.summary()
        print('episode {}'.format(episode_num))
        break
