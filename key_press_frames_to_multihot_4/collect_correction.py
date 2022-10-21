import os
from time import time

import numpy as np

from collect_initial import DataCollector as InitialDataCollector

correction_images_fn = 'correction_images_uint8.npy'
correction_key_scores_fn = 'correction_key_scores_float32.npy'
correction_prev_key_states_fn = 'correction_prev_key_states_float32.npy'
correction_labels_fn = 'correction_labels_float32.npy'


class DataCollector(InitialDataCollector):

    def __init__(self, array_length):
        super().__init__(array_length)

    def save(self):
        images_save_fn = correction_images_fn
        key_scores_save_fn = correction_key_scores_fn
        prev_key_states_save_fn = correction_prev_key_states_fn
        labels_save_fn = correction_labels_fn

        if os.path.isfile(correction_images_fn):
            print('Correction data already exists. Appending timestamp to filenames.')
            append_string = '_' + str(time())
            images_save_fn += append_string
            key_scores_save_fn += append_string
            prev_key_states_save_fn += append_string
            labels_save_fn += append_string

        print('Saving {} frames'.format(self.index))

        self.images = self.images[:self.index]
        np.save(images_save_fn, self.images)

        self.key_scores = self.key_scores[:self.index]
        np.save(key_scores_save_fn, self.key_scores)

        self.prev_key_states = self.prev_key_states[:self.index]
        np.save(prev_key_states_save_fn, self.prev_key_states)

        self.labels = self.labels[:self.index]
        np.save(labels_save_fn, self.labels.astype('float32'))
