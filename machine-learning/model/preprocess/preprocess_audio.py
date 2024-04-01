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
                 sample_rate = 22050,
                 bins_per_octave = 36,
                 hop_length = 512,
                 octaves=8,
                 start_note='',
                 use_hcqt=False):

        self.audio_path = audio_path
        self.annotation_path = annotation_path
        self.save_path = save_path
        self.sample_rate = sample_rate
        self.octaves = octaves
        self.start_note = start_note
        self.use_hcqt = use_hcqt
        self.bins_per_octave = bins_per_octave
        self.cqt_bins = self.bins_per_octave * self.octaves
        self.hop_length = hop_length

        self.string_midi_pitches = [40, 45, 50, 55, 59, 64] #midi notes of open strings
        self.highest_fret = 19
        self.num_of_classes = self.highest_fret + 2 # string freely played + muted

        self.output = {}

    def preprocess_audio(self, audio):
        audio = audio.astype(float)
        audio = librosa.util.normalize(audio)
        if not self.use_hcqt:
            data = np.abs(librosa.cqt(y=audio,
                                      hop_length=self.hop_length,
                                      sr=self.sample_rate,
                                      n_bins=self.cqt_bins,
                                      bins_per_octave=self.bins_per_octave))
        else:
            fmin = librosa.note_to_hz(self.start_note)
            data = np.abs(librosa.cqt(y=audio,
                                      hop_length=self.hop_length,
                                      sr=self.sample_rate,
                                      n_bins=self.cqt_bins,
                                      bins_per_octave=self.bins_per_octave,
                                      fmin = fmin))
        return data

    def correct_numbering(self, n):
        n += 1
        if n < 0 or n > self.highest_fret + 1:
            n = 0
        return n

    def one_hot_encode(self, label):
        return to_categorical(label, self.num_of_classes)

    def process_label(self, label):
        label = [self.correct_numbering(n) for n in label]
        return self.one_hot_encode(label)

    def process_labels(self, labels):
        return np.array([self.process_label(label) for label in labels])


    def load_data_from_file(self, filename):
        audio_file = self.audio_path + filename + '_mic.wav'
        annotation_file = self.annotation_path + filename + '.jams'
        jam = jams.load(annotation_file)
        data, _ = librosa.load(audio_file, sr=self.sample_rate)

        self.output["spec"] = np.swapaxes(self.preprocess_audio(data), 0, 1)

        frame_indices = range(len(self.output["spec"]))
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
        self.output["tab"] = self.process_labels(labels)
        return len(self.output["tab"])

    def save_archive(self, filename):
        np.savez(filename, **self.output)

    def preprocess_data(self):
        for annotation_file in os.listdir(self.annotation_path):
            filename = annotation_file[:-5]
            num_frames = self.load_data_from_file(filename)
            print("done: " + filename + ", " + str(num_frames) + " frames")
            save_path = self.save_path
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            self.save_archive(save_path + filename + ".npz")


if __name__ == '__main__':
    generator = PreprocessGenerator(audio_path='data/audio/GuitarSet/audio/',
                                    annotation_path='data/audio/GuitarSet/annotation/',
                                    save_path='data/archived/GuitarSet/5_octaves/',
                                    octaves=5,
                                    use_hcqt=True,
                                    start_note='C2'
                                    )
    generator.preprocess_data()