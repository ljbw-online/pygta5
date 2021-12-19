import os
from common import EPOCHS, MODEL_NAME
from network import model
from prepare_data import training_inputs, training_outputs, validation_inputs, validation_outputs

if os.path.isfile('{}.index'.format(MODEL_NAME)):
    print('Loading previous model')
    model.load(MODEL_NAME)

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

