import os.path
from typing import Tuple, List

from fastapi import UploadFile
import librosa
from scipy.io.wavfile import write
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

    def preprocess_audio(self, file : UploadFile) -> str:
        if not file.filename.endswith('.wav'):
            raise InvalidFileFormatException('Invalid file format')
        folder_name = file.filename.replace('.wav', '')
        # read file
        y, sr = self.__read_audio(file)
        # split segments
        segments = [] #self.__onset_detection(y=y, sr=sr)
        # write to temp folder
        self.__save_temp_audio_files(sr=sr, segments=segments, folder_name=folder_name)
        #save full audio
        write(f"../data/audio/{folder_name}/audio.wav", sr, y)
        # archive files
        self.__archive_files(folder_name=folder_name)
        #archive full audio
        self.__archive_file(f"../data/audio/{folder_name}/audio.wav",
                            f"../data/archived/{folder_name}/audio.npz")
        return folder_name


    def __read_audio(self, file : UploadFile) -> Tuple[np.ndarray, float]:
        return librosa.load(file.file, sr=self.sr)

    #todo tune this function
    def __onset_detection(self, y : np.ndarray, sr : float) -> List[np.ndarray]:

        y = librosa.util.normalize(y)
        onset_frames = librosa.onset.onset_detect(
            y = y,
            sr = sr,
            wait = 1,
            pre_avg = 1,
            post_avg = 1,
            pre_max = 1,
            post_max = 1,
            backtrack = True,
            normalize = True
        )
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        start_time = 0
        end_time = (1 / sr) * len(y)
        onset_times = np.insert(onset_times, 0, start_time)
        onset_times = np.append(onset_times, end_time)

        segments = []
        for i in range(len(onset_times) - 1):
            start_time = onset_times[i]
            end_time = onset_times[i + 1]
            if (end_time - start_time) * sr:
                segment = y[int(start_time * sr):int(end_time * sr)]
                if max(segment) > 0.3 and (len(segment) / sr) > 0.05:
                    segments.append(segment)


        return segments

    def __generate_index(self, i : int) -> str:
        str_i = str(i)
        result = '0' * (4 - len(str_i))
        result += str_i
        return result

    def __save_temp_audio_files(self, sr, segments : List[np.ndarray], folder_name : str):
        directory_path = f"../data/audio/{folder_name}"
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        # for i, segment in enumerate(segments):
        #     write(f"{directory_path}/temp_{self.__generate_index(i+1)}.wav", sr, segment)
        pass

    def __archive_files(self, folder_name : str):

        archive_path = f"../data/archived/{folder_name}"
        audio_path = f"../data/audio/{folder_name}"

        if not os.path.exists(archive_path):
            os.makedirs(archive_path)
        #
        # for audio_file in os.listdir(audio_path):
        #     npz_file = audio_file.replace('.wav', '.npz')
        #     self.__archive_file(os.path.join(audio_path, audio_file), os.path.join(archive_path, npz_file))
        pass

    def __archive_file(self, audio_file_path : str, archive_file_path : str):
        output = {}
        audio, _ = librosa.load(audio_file_path, sr=self.sr)
        audio = audio.astype(float)
        audio = librosa.util.normalize(audio)
        audio = np.abs(librosa.cqt(audio, hop_length=self.hop_length, sr=self.sr, n_bins=self.cqt_n_bins, bins_per_octave=self.cqt_n_bins_per_octave))
        audio = np.swapaxes(audio, 0, 1)
        output["representation"] = audio
        np.savez(archive_file_path, **output)