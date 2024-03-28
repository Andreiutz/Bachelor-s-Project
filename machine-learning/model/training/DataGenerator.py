import numpy as np
import keras
import pandas as pd

class DataGenerator(keras.utils.Sequence):

    def __init__(self,
                 list_IDs,
                 data_path="data/archived/GuitarSet/",
                 batch_size=128,
                 shuffle=True,
                 con_win_size=9
                 ):

        self.list_IDs = list_IDs
        self.data_path = data_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.con_win_size = con_win_size
        self.halfwin = con_win_size // 2

        self.batch_input = []
        self.batch_output = {
            "EString": [],
            "AString": [],
            "DString": [],
            "GString": [],
            "BString": [],
            "eString": [],
        }

        self.on_epoch_end()

    def __len__(self):
        # number of batches per epoch
        return int(np.floor(float(len(self.list_IDs)) / self.batch_size))

    def __getitem__(self, index):
        # generate indices of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # generate data
        X, y = self.__data_generation(list_IDs_temp)
        return np.array(X), {k: np.array(v) for k, v in y.items()}


    def on_epoch_end(self):
        # Updates indexes after each epoch
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def custom_pad(self, array, pad_width, iaxis, kwargs):
        # pad_width will be a tuple of tuples ((before_1, after_1), (before_2, after_2), ...)
        # For your 2D array, you're interested in pad_width[0] as you're padding along axis 0
        pad_before = array[0, :]  # First row
        pad_after = array[-1, :]  # Last row

        # Repeat the first and last row self.halfwin times
        array = np.concatenate(
            [np.tile(pad_before, (pad_width[0][0], 1)), array, np.tile(pad_after, (pad_width[0][1], 1))], axis=0)
        return array

    def __data_generation(self, list_IDs_temp):
        # Generates data containing batch_size samples
        # X : (n_samples, *dim, n_channels)

        # Initialization
        self.batch_input = []
        self.batch_output = {
            "EString": [],
            "AString": [],
            "DString": [],
            "GString": [],
            "BString": [],
            "eString": [],
        }
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            data_dir = self.data_path
            filename = "_".join(ID.split("_")[:-1]) + ".npz"
            frame_idx = int(ID.split("_")[-1])

            loaded = np.load(data_dir + filename)

            original_data = loaded["spec"]
            first_row = original_data[0, :].reshape(1, -1)
            last_row = original_data[-1, :].reshape(1, -1)
            repeated_first_row = np.repeat(first_row, self.halfwin, axis=0)
            repeated_last_row = np.repeat(last_row, self.halfwin, axis=0)
            full_x = np.concatenate([repeated_first_row, original_data, repeated_last_row], axis=0)

            sample_x = full_x[frame_idx: frame_idx + self.con_win_size]
            self.batch_input.append(np.expand_dims(np.swapaxes(sample_x, 0, 1), -1))

            # Store label
            output = loaded["tab"][frame_idx]
            self.batch_output["EString"].append(output[0])
            self.batch_output["AString"].append(output[1])
            self.batch_output["DString"].append(output[2])
            self.batch_output["GString"].append(output[3])
            self.batch_output["BString"].append(output[4])
            self.batch_output["eString"].append(output[5])


        return self.batch_input, self.batch_output
