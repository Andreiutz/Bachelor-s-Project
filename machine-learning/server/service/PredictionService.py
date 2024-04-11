import os
import sys

import tensorflow as tf
from server.service.utils import *
from server.exceptions.FileNotFoundException import FileNotFoundException
from server.exceptions.PathNotFoundException import PathNotFoundException

SECONDS_PER_FRAME = 1 / (22050 / 512)
DATA_PATH = "../data"

class PredictionService:
    def __init__(self, preprocess_service, cache_service):
        self.__preprocess_service = preprocess_service
        self.__cache_service = cache_service
        self.bins_per_octave = 36
        self.frame_size = 9
        self.model_path = "model/training/saved/2024-04-03/2_1_8_octaves/model/"
        self.model = tf.keras.models.load_model(self.model_path)

    # uses onset detection to determine each strum and computes the average for each interval
    def predict_tablature(self, folder_name, load=False, cache=True) -> dict:
        folder_path = f"{DATA_PATH}/{folder_name}"
        if load:
            try:
                return self.__cache_service.get_data(f"{folder_path}/tabs.json")
            except PathNotFoundException:
                pass

        predictions = {
            "frames": [],
            "times": [],
            "strum": []
        }
        archive_path = f"{folder_path}/file.npz"
        audio_path = f"{folder_path}/file.wav"

        if not os.path.exists(archive_path) or not os.path.exists(audio_path):
            raise FileNotFoundException('File not found')

        result = self.__predict_full_audio(archive_path)
        y, sr = librosa.load(audio_path)
        y = librosa.util.normalize(y)
        onset_frames_pairs = get_onset_frames(y, sr)

        for frame_pair in onset_frames_pairs:
            predicted_strum = get_predicted_strum_of_batch(result, frame_pair[0], frame_pair[1])
            predictions["frames"].append([int(frame_pair[0]), int(frame_pair[1])])
            predictions["times"].append(
                [int(frame_pair[0]) * SECONDS_PER_FRAME, int(frame_pair[1]) * SECONDS_PER_FRAME])
            predictions["strum"].append(predicted_strum)
        if cache:
            self.__cache_service.cache_data(path=folder_path, data=predictions, file_name="tabs")
        return predictions

    # returns all the strums detected, merged together if 2 identic are adjacent
    # also returns start and end times and frames for each batch of identic strums
    def predict_full_samples(self, folder_name, load=False, cache=True) -> dict:
        folder_path = f"{DATA_PATH}/{folder_name}"
        try:
            return self.__cache_service.get_data(f"{folder_path}/full.json")
        except PathNotFoundException:
            pass
        predictions = {
            "frames" : [],
            "times" : [],
            "strum" : []
        }
        archive_path = f"{folder_path}/file.npz"
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
        if cache:
            self.__cache_service.cache_data(path=folder_path, data=predictions, file_name="full")
        return predictions

    def __predict_full_audio(self, path_to_audio) -> np.ndarray:
        x = np.load(path_to_audio)
        frames = len(x["representation"])
        full_x = x["representation"]
        X = np.empty((frames, self.bins_per_octave * 8, self.frame_size, 1))
        for i in range(frames - self.frame_size):
            sample = full_x[i:i + self.frame_size]
            X[i,] = np.expand_dims(np.swapaxes(sample, 0, 1), -1)
        return self.model.predict(X)
