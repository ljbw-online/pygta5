import numpy as np
from common import INPUT_WIDTH,INPUT_HEIGHT,OUTPUT_SHAPE,RESIZE_WIDTH,RESIZE_HEIGHT

training_data = np.load('training_data.npy')
new_training_data = np.empty((len(training_data),INPUT_HEIGHT,INPUT_WIDTH))
print(training_data.shape,new_training_data.shape)
# multihot_array = training_data[:, -4:, -1]
# onehot_outputs = np.empty((len(training_data),9))

mh_w = np.array([1,0,0,0],dtype='uint8')
mh_a = np.array([0,1,0,0],dtype='uint8')
mh_s = np.array([0,0,1,0],dtype='uint8')
mh_d = np.array([0,0,0,1],dtype='uint8')
mh_wa = np.array([1,1,0,0],dtype='uint8')
mh_wd = np.array([1,0,0,1],dtype='uint8')
mh_sa = np.array([0,1,1,0],dtype='uint8')
mh_sd = np.array([0,0,1,1],dtype='uint8')
mh_nk = np.array([0,0,0,0],dtype='uint8')

w = np.array([1,0,0,0,0,0,0,0,0],dtype='uint8')
a = np.array([0,1,0,0,0,0,0,0,0],dtype='uint8')
s = np.array([0,0,1,0,0,0,0,0,0],dtype='uint8')
d = np.array([0,0,0,1,0,0,0,0,0],dtype='uint8')
wa = np.array([0,0,0,0,1,0,0,0,0],dtype='uint8')
wd = np.array([0,0,0,0,0,1,0,0,0],dtype='uint8')
sa = np.array([0,0,0,0,0,0,1,0,0],dtype='uint8')
sd = np.array([0,0,0,0,0,0,0,1,0],dtype='uint8')
nk = np.array([0,0,0,0,0,0,0,0,1],dtype='uint8')

def mh_pixels_to_oh_pixels(multihot_array):
    # Division converts it to float64 but the comparison still works
    multihot_array = multihot_array / 255
    if np.array_equal(multihot_array, mh_w):
        return w * 255
    elif np.array_equal(multihot_array, mh_a):
        return a * 255
    elif np.array_equal(multihot_array, mh_s):
        return s * 255
    elif np.array_equal(multihot_array, mh_d):
        return d * 255
    elif np.array_equal(multihot_array, mh_wa):
        return wa * 255
    elif np.array_equal(multihot_array, mh_wd):
        return wd * 255
    elif np.array_equal(multihot_array, mh_sa):
        return sa * 255
    elif np.array_equal(multihot_array, mh_sd):
        return sd * 255
    elif np.array_equal(multihot_array, mh_nk):
        return nk * 255
    else:
        print('Didn\'t match',multihot_array)

# np.save('onehot_outputs.npy',onehot_outputs)

for i in range(len(training_data)):
    print(new_training_data[i,:RESIZE_HEIGHT,:].shape,training_data[i,:RESIZE_HEIGHT,:].shape)
    new_training_data[i,:RESIZE_HEIGHT,:] = training_data[i,:RESIZE_HEIGHT,:]
    for j in range(INPUT_WIDTH):
        print(new_training_data[i,(RESIZE_HEIGHT):,j].shape,mh_pixels_to_oh_pixels(training_data[i, -4:, j]).shape)
        new_training_data[i,(RESIZE_HEIGHT):,j] = mh_pixels_to_oh_pixels(training_data[i, -4:, j])