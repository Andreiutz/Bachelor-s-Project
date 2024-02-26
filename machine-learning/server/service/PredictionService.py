import os
from typing import List

import librosa
import numpy as np
import tensorflow as tf
from server.service.utils import *

SECONDS_PER_FRAME = 1 / (22050 / 512)

class PredictionService:
    def __init__(self, preprocess_service):
        self.__preprocess_service = preprocess_service
        self.bins_per_octave = 36
        self.con_win_size = 9
        self.model_path = "model/training/saved/2024-02-22/2_1/model"
        self.model = tf.keras.models.load_model(self.model_path)

    # uses onset detection to determine each strum and computes the average for each interval
    def predict_strums(self, archive_folder_name) -> dict:
        predictions = {
            "frames" : [],
            "times" : [],
            "strum" : []
        }
        archive_path = "../data/archived/" + archive_folder_name + "/audio.npz"
        audio_path = "../data/audio/" + archive_folder_name + "/audio.wav"
        result = self.__predict_full_audio(archive_path)
        y, sr = librosa.load(audio_path)
        y = librosa.util.normalize(y)
        onset_frames_pairs = get_onset_frames(y, sr)

        for frame_pair in onset_frames_pairs:
            predicted_strum = get_predicted_strum_of_batch(result, frame_pair[0], frame_pair[1])
            predictions["frames"].append([int(frame_pair[0]), int(frame_pair[1])])
            predictions["times"].append([int(frame_pair[0]) * SECONDS_PER_FRAME, int(frame_pair[1]) * SECONDS_PER_FRAME])
            predictions["strum"].append(predicted_strum)
        return predictions

    # returns all the strums detected, merged together if 2 identic are adjacent
    # also returns start and end times and frames for each batch of identic strums
    def predict_all(self, archive_folder_name) -> dict:
        predictions = {
            "frames" : [],
            "times" : [],
            "strum" : []
        }
        archive_path = "../data/archived/" + archive_folder_name + "/audio.npz"
        result = [get_predicted_strum_of_sample(x) for x in np.swapaxes(self.__predict_full_audio(archive_path), 0, 1)]
        result.append([-2 for i in range(6)])
        index = 0
        start = 0
        while index < len(result)-1:
            if result[index] != result[index + 1]:
                predictions["frames"].append([start, index])
                predictions["times"].append([start * SECONDS_PER_FRAME, index * SECONDS_PER_FRAME])
                predictions["strum"].append(result[index])
                start = index+1
            index += 1
        return predictions

    def __predict_sample(self, path_to_sample) -> List[int]:
        x = np.load(path_to_sample)
        frames = len(x["representation"])
        full_x = x["representation"]
        X = np.empty((frames, self.bins_per_octave * 8, self.con_win_size, 1))
        for i in range(frames - self.con_win_size):
            sample = full_x[i:i + self.con_win_size]
            X[i,] = np.expand_dims(np.swapaxes(sample, 0, 1), -1)
        y = self.model.predict_strums(X)
        strum = get_predicted_strum_of_batch(y)
        return strum

    def __predict_full_audio(self, path_to_audio) -> np.ndarray:
        x = np.load(path_to_audio)
        frames = len(x["representation"])
        full_x = x["representation"]
        X = np.empty((frames, self.bins_per_octave * 8, self.con_win_size, 1))
        for i in range(frames - self.con_win_size):
            sample = full_x[i:i + self.con_win_size]
            X[i,] = np.expand_dims(np.swapaxes(sample, 0, 1), -1)
        return self.model.predict(X)