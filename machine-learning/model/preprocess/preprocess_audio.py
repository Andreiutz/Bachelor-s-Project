import jams
import os
import numpy as np
import librosa
from keras.utils import to_categorical
from scipy.io import wavfile

class PreprocessGenerator:

    def __init__(self,
                 audio_path,
                 annotation_path,
                 save_path,
                 normalize = True,
                 sample_rate = 22050,
                 cqt_bins_per_octave = 36,
                 n_fft = 2048,
                 hop_length = 512):

        self.audio_path = audio_path
        self.annotation_path = annotation_path
        self.save_path = save_path
        self.normalize = normalize
        self.sample_rate = sample_rate
        self.cqt_bins_per_octave = cqt_bins_per_octave
        self.cqt_bins = self.cqt_bins_per_octave * 8 # since there are 8 octaves to be concerned of
        self.n_fft = n_fft
        self.hop_length = hop_length

        self.string_midi_pitches = [40, 45, 50, 55, 59, 64]
        self.highest_fret = 19
        self.num_of_classes = self.highest_fret + 2 # string freely played + muted

        self.output = {}

    def preprocess_audio(self, audio):
        audio = audio.astype(float)
        if self.normalize:
            audio = librosa.util.normalize(audio)
        data = np.abs(librosa.cqt(y=audio,
                                  hop_length=self.hop_length,
                                  sr = self.sample_rate,
                                  n_bins=self.cqt_bins,
                                  bins_per_octave=self.cqt_bins_per_octave))
        return data

    def correct_numbering(self, n):
        n += 1
        if n < 0 or n > self.highest_fret:
            n = 0
        return n

    def categorical(self, label):
        return to_categorical(label, self.num_of_classes)

    def clean_label(self, label):
        label = [self.correct_numbering(n) for n in label]
        return self.categorical(label)

    def clean_labels(self, labels):
        return np.array([self.clean_label(label) for label in labels])


    def load_data_for_file(self, filename):
        audio_file = self.audio_path + filename + '_mic.wav'
        annotation_file = self.annotation_path + filename + '.jams'
        jam = jams.load(annotation_file)
        data, _ = librosa.load(audio_file, sr=self.sample_rate)

        self.output["representation"] = np.swapaxes(self.preprocess_audio(data), 0, 1)

        frame_indices = range(len(self.output["representation"]))
        times = librosa.frames_to_time(frame_indices, sr=self.sample_rate, hop_length=self.hop_length)

        labels = []
        for string_index in range(6):
            string_annotation = jam.annotations["note_midi"][string_index]
            string_label_samples = string_annotation.to_samples(times)
            for i in frame_indices:
                if not string_label_samples[i]:
                    string_label_samples[i] = -1
                else:
                    string_label_samples[i] = int(round(string_label_samples[i][0]) - self.string_midi_pitches[string_index])
            labels.append([string_label_samples])

        labels = np.array(labels)
        labels = np.swapaxes(np.squeeze(labels), 0, 1)
        self.output["labels"] = self.clean_labels(labels)
        return len(self.output["labels"])

    def save_archive(self, filename):
        np.savez(filename, **self.output)

    def get_nth_filename(self, n):
        filenames = np.sort(np.array(os.listdir(self.annotation_path)))
        return filenames[n][:-5]

    def compute_data(self, n):
        for i in range(n):
            filename = self.get_nth_filename(i)
            num_frames = self.load_data_for_file(filename)
            print("done: " + filename + ", " + str(num_frames) + " frames")
            save_path = self.save_path
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            self.save_archive(save_path + filename + ".npz")


if __name__ == '__main__':
    generator = PreprocessGenerator(audio_path='../../data/audio/GuitarSet/audio/',
                                    annotation_path='../../data/audio/GuitarSet/annotation/',
                                    save_path='../../data/archived/GuitarSet/')
    generator.compute_data(360) #there are 360 audio files