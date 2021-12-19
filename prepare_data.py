import numpy as np
import os

from common import (DATA_FILE_NAME, OUTPUT_LENGTH,
    w,a,s,d,wa,wd,sa,sd,nk, INPUT_WIDTH, INPUT_HEIGHT)

# import sys
# sys.path.append('C:/Users/ljbw/PycharmProjects/pygta5')
# import importlib
# importlib.reload(common)

def stats(data):
    forwards = 0
    lefts = 0
    brakes = 0
    rights = 0
    forward_lefts = 0
    forward_rights = 0
    brake_lefts = 0
    brake_rights = 0
    nokeys = 0

    for keystate in data[:,-1,:OUTPUT_LENGTH]:
        if np.array_equal(keystate,w):
            forwards += 1
        elif np.array_equal(keystate,a):
            lefts += 1
        elif np.array_equal(keystate,s):
            brakes += 1
        elif np.array_equal(keystate,d):
            rights += 1
        elif np.array_equal(keystate,wa):
            forward_lefts += 1
        elif np.array_equal(keystate,wd):
            forward_rights += 1
        elif np.array_equal(keystate,sa):
            brake_lefts += 1
        elif np.array_equal(keystate,sd):
            brake_rights += 1
        elif np.array_equal(keystate,nk):
            nokeys += 1
        else:
            print('huh?',keystate)

    print('forwards',forwards)
    print('lefts',lefts)
    print('brakes',brakes)
    print('rights',rights)
    print('forward_lefts',forward_lefts)
    print('forward_rights',forward_rights)
    print('brake_lefts',brake_lefts)
    print('brake_rights',brake_rights)
    print('nokeys',nokeys)

    return [forwards,lefts,brakes,lefts,forward_lefts,forward_rights,brake_lefts,brake_rights,nokeys]

if os.path.isfile('training_inputs.npy'):
    training_inputs = np.load('training_inputs.npy')
    training_outputs = np.load('training_outputs.npy')
    validation_inputs = np.load('validation_inputs.npy')
    validation_outputs = np.load('validation_outputs.npy')
else:
    collected_data = np.load(DATA_FILE_NAME)
    [forwards,lefts,brakes,rights,forward_lefts,forward_rights,brake_lefts,brake_rights,nokeys] = stats(collected_data)

    # Shift outputs along by 4 so that the model is predicting what the keypress should be four
    # frames in the future. This reflects the fact that a human presses the key that corresponds to
    # the frame they saw more-or-less a fifth of a second ago.
    collected_data[:,-1] = np.roll(collected_data[:,-1],-4,axis=0)
    collected_data = collected_data[4:] # Get rid of the first four frames which now have incorrect output_rows

    # Balance data
    np.random.shuffle(collected_data) # So that we get forwards frames from all over the data
    forwards_frames = np.empty((forwards,INPUT_HEIGHT + 1, INPUT_WIDTH),dtype='uint8')
    ff_counter = 0
    non_forwards_frames = np.empty(((len(collected_data) - forwards),INPUT_HEIGHT + 1, INPUT_WIDTH),dtype='uint8')
    nff_counter = 0

    for frame in collected_data:
        if np.array_equal(frame[-1,:OUTPUT_LENGTH],w):
            forwards_frames[ff_counter] = frame
            ff_counter += 1
        else:
            non_forwards_frames[nff_counter] = frame
            nff_counter += 1

    balanced_data = np.concatenate((forwards_frames[:round((forward_lefts + forward_rights)/2)], non_forwards_frames))
    # print(len(forwards_frames),len(non_forwards_frames),len(balanced_data))
    np.random.shuffle(balanced_data) # Because the above operation resulted in all forwards followed by all non_forwards

    # Split data
    validation_index = round(len(balanced_data) * 0.1)
    training_data = balanced_data[validation_index:]
    validation_data = balanced_data[:validation_index]

    training_inputs = training_data[:, :-1]
    training_outputs = training_data[:, -1, :OUTPUT_LENGTH]

    validation_inputs = validation_data[:, :-1]
    validation_outputs = validation_data[:, -1, :OUTPUT_LENGTH]

    training_inputs = training_inputs.reshape(-1, INPUT_WIDTH, INPUT_HEIGHT, 1)
    validation_inputs = validation_inputs.reshape(-1, INPUT_WIDTH, INPUT_HEIGHT, 1)

    np.save('training_inputs',training_inputs)
    np.save('training_outputs',training_outputs)
    np.save('validation_inputs',validation_inputs)
    np.save('validation_outputs',validation_outputs)
