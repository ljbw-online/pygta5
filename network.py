""" AlexNet.
References:
    - Alex Krizhevsky, Ilya Sutskever & Geoffrey E. Hinton. ImageNet
    Classification with Deep Convolutional Neural Networks. NIPS, 2012.
Links:
    - [AlexNet Paper](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
"""

# This isn't in common.py because tensorflow takes ages to import and prints loads of warnings.

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import local_response_normalization

def alexnet(input_width, input_height, lr, output_length):
    network = input_data(shape=[None, input_width, input_height, 1], name='input')
    network = conv_2d(network, 96, 11, strides=4, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 256, 5, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 256, 3, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, output_length, activation='softmax')
    network = regression(
        network,
        optimizer='adam',
        loss='categorical_crossentropy',
        learning_rate=lr,
        name='targets'
    )
    model = tflearn.DNN(
        network,
        # checkpoint_path='model_alexnet',
        # max_checkpoints=1,
        # tensorboard_verbose=2,
        # tensorboard_dir='log'
    )
    return model

from common import INPUT_WIDTH, INPUT_HEIGHT, LR, OUTPUT_LENGTH
model = alexnet(INPUT_WIDTH, INPUT_HEIGHT, LR, OUTPUT_LENGTH)