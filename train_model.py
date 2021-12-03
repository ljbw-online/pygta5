import numpy as np
import os
from common import INPUT_WIDTH, INPUT_HEIGHT, EPOCHS, MODEL_NAME, model, DATA_FILE_NAME, OUTPUT_LENGTH

if os.path.isfile('{}.index'.format(MODEL_NAME)):
    print('Loading previous model')
    model.load(MODEL_NAME)

if os.path.isfile('training_inputs.npy'):
    training_inputs = np.load('training_inputs.npy')
    training_outputs = np.load('training_outputs.npy')
    validation_inputs = np.load('validation_inputs.npy')
    validation_outputs = np.load('validation_outputs.npy')
else:
    data = np.load(DATA_FILE_NAME)

    # Shift outputs along by 4 so that the model is predicting what the keypress should be four
    # frames in the future. This reflects the fact that a human presses the key that corresponds to
    # the frame they saw about a fifth of a second ago.
    data[:,-1] = np.roll(data[:,-1],-4,axis=0)
    data = data[4:] # Get rid of the first four frames which now have incorrect output_rows
    np.random.shuffle(data)
    validation_index = round(len(data) * 0.1)
    training_inputs = data[validation_index:]
    validation_inputs = data[:validation_index]

    training_outputs = training_inputs[:, -1, :OUTPUT_LENGTH]
    training_inputs = training_inputs[:, :-1]

    validation_outputs = validation_inputs[:, -1, :OUTPUT_LENGTH]
    validation_inputs = validation_inputs[:, :-1]

    training_inputs = training_inputs.reshape(-1, INPUT_WIDTH, INPUT_HEIGHT, 1)
    validation_inputs = validation_inputs.reshape(-1, INPUT_WIDTH, INPUT_HEIGHT, 1)

# print(validation_outputs.shape,validation_inputs.shape,training_inputs.shape)
model.fit(
      {'input': training_inputs}
    , {'targets': training_outputs}
    , n_epoch=EPOCHS
    , validation_set=({'input': validation_inputs}, {'targets': validation_outputs})
    , snapshot_step=500
    , show_metric=True
    , run_id=MODEL_NAME
    , batch_size=256
)

model.save(MODEL_NAME)

# tensorboard --logdir="foo:C:/Users/ljbw/Documents/Python/.venv/pygta5/Tutorial Codes/Part 8-13 code/log/Airtug"

# These will keep being used even if more training data has been added. Will just have new validation data everytime
# for now.
# if not os.path.isfile('training_inputs.npy'):
#     # Save here so that the same validation set is used across multiple runs of this script
#     # This is at the end so that we only save if the rest of the file succeeds. Otherwise
#     # I have to keep remembering to delete the possibly-bad npy files that were created.
#     np.save('training_inputs.npy', training_inputs)
#     np.save('training_outputs.npy', training_outputs)
#     np.save('validation_inputs.npy', validation_inputs)
#     np.save('validation_outputs.npy', validation_outputs)
