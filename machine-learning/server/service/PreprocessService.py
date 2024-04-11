import os.path
from typing import Tuple, List

from fastapi import UploadFile
import librosa
import numpy as np
from server.exceptions.InvalidFIleFormatException import InvalidFileFormatException
import warnings
warnings.filterwarnings("ignore")

class PreprocessService:

    def __init__(self,
                 sr = 22050,
                 cqt_n_bins_per_octave=36,
                 hop_length = 512):
        self.sr = sr
        self.cqt_n_bins_per_octave = cqt_n_bins_per_octave
        self.cqt_n_bins = cqt_n_bins_per_octave * 8 # create cqt image from 8 octaves
        self.hop_length = hop_length

    def archive_file_from_upload(self, file : UploadFile, folder_name : str):
        if not file.filename.endswith('.wav'):
            raise InvalidFileFormatException('Invalid file format')

        archive_path = f"../data/{folder_name}/"
        self.__make_dir(archive_path)
        self.__archive_file(f"../data/{folder_name}/")

    def archive_file_from_folder(self, folder_name : str):
        archive_path = f"../data/{folder_name}/"

        self.__make_dir(archive_path)
        self.__archive_file(f"../data/{folder_name}/")

    def __archive_file(self, folder_path : str):
        audio_file_path = folder_path + "file.wav"
        archive_file_path = folder_path + "file.npz"
        output = {}
        audio, _ = librosa.load(audio_file_path, sr=self.sr)
        audio = audio.astype(float)
        audio = librosa.util.normalize(audio)
        audio = np.abs(librosa.cqt(audio, hop_length=self.hop_length, sr=self.sr, n_bins=self.cqt_n_bins, bins_per_octave=self.cqt_n_bins_per_octave))
        audio = np.swapaxes(audio, 0, 1)
        output["representation"] = audio
        np.savez(archive_file_path, **output)

    def __make_dir(self, path : str) -> bool:
        if not os.path.exists(path):
            os.makedirs(path)
            return True
        return False