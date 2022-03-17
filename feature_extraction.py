import numpy as np
import cv2
from time import sleep
from tensorflow.keras.models import load_model, Model

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# model = load_model('Taxi')
#
# # feature extraction model
# fem = Model(inputs=model.input, outputs=[model.layers[2].output])

fem = load_model('coast_brake_feature')

test_images = np.load('test_images.npy')

for test_image in test_images:
    features = fem(test_image.reshape((1, 90, 160, 3)))[0].numpy()

    # print(np.max(features[:, :, :3]))
    # print(features.shape)
    features_max = np.max(features)
    features_min = np.min(features)

    features = (features - features_min) / (features_max - features_min)


    # feature_img1 = (features[:, :, :3] * 255).astype('uint8')
    # # feature_img2 = (features[:, :, 3:] * 255).astype('uint8')
    #
    # img0 = feature_img1[:, :, 0]
    # img1 = feature_img1[:, :, 1]
    # img2 = feature_img1[:, :, 2]
    #
    # test_image = cv2.resize(test_image, (960, 540), interpolation=cv2.INTER_NEAREST)
    # # feature_img1 = cv2.resize(feature_img1, (960, 540), interpolation=cv2.INTER_NEAREST)
    # # feature_img2 = cv2.resize(feature_img2, (960, 540), interpolation=cv2.INTER_NEAREST)
    #
    # img0 = cv2.resize(img0, (960, 540), interpolation=cv2.INTER_NEAREST)
    # img1 = cv2.resize(img1, (960, 540), interpolation=cv2.INTER_NEAREST)
    # img2 = cv2.resize(img2, (960, 540), interpolation=cv2.INTER_NEAREST)
    #
    # cv2.imshow('test_image', test_image)
    # # cv2.imshow('feature_img1', feature_img1)
    # # cv2.imshow('feature_img2', feature_img2)
    # cv2.imshow('img0', img0)
    # cv2.imshow('img1', img1)
    # cv2.imshow('img2', img2)

    features = cv2.resize(features, (960, 540), interpolation=cv2.INTER_NEAREST)
    cv2.imshow('coast_brake_features', features)

    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

    # input('a')
