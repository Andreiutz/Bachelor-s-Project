import os
from typing import List
import numpy as np
import tensorflow as tf

class PredictionService:
    def __init__(self, preprocess_service):
        self.__preprocess_service = preprocess_service
        self.bins_per_octave = 36
        self.con_win_size = 9
        self.model_path = "../model/training/saved/2024-02-22/2_1/model"
        self.model = tf.keras.models.load_model(self.model_path)

    def predict(self, archive_folder_name) -> List[List[int]]:
        archive_path = "../data/archived/temp/" + archive_folder_name + "/audio.npz"
        result = self.__predict_full_audio(archive_path)
        # for sample in os.listdir(archive_path):
        #     print(self.__predict_sample(archive_path + sample))
        return []

    def __predict_sample(self, path_to_sample) -> List[int]:
        x = np.load(path_to_sample)
        frames = len(x["representation"])
        full_x = x["representation"]
        X = np.empty((frames, self.bins_per_octave * 8, self.con_win_size, 1))
        for i in range(frames - self.con_win_size):
            sample = full_x[i:i + self.con_win_size]
            X[i,] = np.expand_dims(np.swapaxes(sample, 0, 1), -1)
        y = self.model.predict(X)
        strum = self.__get_predicted_strum_of_batch(y)
        return strum

    def __predict_full_audio(self, path_to_audio) -> List[List[int]]:
        x = np.load(path_to_audio)
        frames = len(x["representation"])
        full_x = x["representation"]
        X = np.empty((frames, self.bins_per_octave * 8, self.con_win_size, 1))
        for i in range(frames - self.con_win_size):
            sample = full_x[i:i + self.con_win_size]
            X[i,] = np.expand_dims(np.swapaxes(sample, 0, 1), -1)
        y = self.model.predict(X)
        return [self.__get_predicted_strum_of_sample(x) for x in np.swapaxes(y, 0, 1)]

    #given a set of samples which were determined to be part of same note, return the strum
    def __get_predicted_strum_of_batch(self, y):
        result = []
        for string in y:
            summed_arr = np.sum(string, axis=0)
            result.append(np.argmax(summed_arr) - 1)
        return result

    #returns the strum of a single sample (1 / 43) of a second
    def __get_predicted_strum_of_sample(self, y):
        result = []
        for string in y:
            result.append(np.argmax(string)-1)
        return result