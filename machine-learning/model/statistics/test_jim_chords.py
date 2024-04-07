import sys

import tensorflow as tf
import os
import numpy as np
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
os.chdir(project_root)
chord_labels = ["a", "am", "bm", "c", "d", "dm", "e", "em", "f", "g"]

a = [-1, 0, 2, 2, 2, 0]
am = [-1, 0, 2, 2, 1, 0]
bm = [-1, 2, 4, 4, 3, 2]
c = [-1, 3, 2, 0, 1, 0]
d = [-1, -1, 0, 2, 3, 2]
dm = [-1, -1, 0, 2, 3, 1]
e = [0, 2, 2, 1, 0, 0]
em = [0, 2, 2, 0, 0, 0]
f = [-1, 3, 3, 2, 1, -1]
g = [3, 2, 0, 0, 3, 3]

pitches = [40,45,50,55,59,64]

chords = {
    "a" : a,
    "am" : am,
    "bm": bm,
    "c" : c,
    "d" : d,
    "dm" : dm,
    "e" : e,
    "em" : em,
    "f" : f,
    "g" : g
}


model_folder = "model/training/saved/2024-04-03/train_83p_1_8_octaves"
model_path = f"{model_folder}/model"
checkpoint_path = f"{model_folder}/checkpoints"
model = tf.keras.models.load_model(model_path)
folder_path = "data/archived/JimChords2012/"
log_file_path = f"{model_folder}/chord_tests_log.txt"
#log_file_path = "saved/tests_log.txt"

final_tab_score = 0
final_pitch_score = 0


def get_predicted_strum(y):
    result = []
    for string in y:
        summed_arr = np.sum(string, axis=0)
        result.append(np.argmax(summed_arr) - 1)
    return result

def get_tab_score(pattern, predict):
    score = 0
    for i in range(len(pattern)):
        if pattern[i] == predict[i]:
            score += 1
    return score

def get_pitch_score(pattern, predict):
    pch = []
    for pitch in range(len(pitches)):
        if pattern[pitch] != -1:
            pch.append(pitches[pitch] + pattern[pitch])

    score = 0
    for i in range(len(predict)):
        if predict[i] == -1 and pattern[i] == -1: # If the string is muted, the string in pattern should also be muted
            score += 1
        elif predict[i] != -1 and pitches[i] + predict[i] in pch: #otherwise check if maybe the current string plays a pitch not necessary from the same string in pattern
            score += 1
            pch.remove(pitches[i] + predict[i])
    return score

def test_chord(pattern, npz_path):
    X = np.load(npz_path)
    #here "repr" key is used, but it is no longer used for preprocessing audio, only works here
    frames = len(X["repr"])
    full_x = np.pad(X["repr"], [(4, 4), (0, 0)], mode='constant')
    X = np.empty((frames, 288, 9, 1))
    for i in range(frames):
        sample = full_x[i:i + 9]
        X[i,] = np.expand_dims(np.swapaxes(sample, 0, 1), -1)

    y = model.predict(X, verbose=0)
    predict = get_predicted_strum(y)
    return {"tab": get_tab_score(pattern, predict), "pitch": get_pitch_score(pattern, predict)}

def test_folder(folder_name):
    tab_score = 0
    pitch_score = 0
    max_score = 0
    path = folder_path + folder_name + "/"
    for file in os.listdir(path):
        score = test_chord(chords[folder_name], path + file)
        tab_score += score["tab"]
        pitch_score += score["pitch"]
        max_score += 6
    tab_acc = tab_score / max_score
    pitch_acc = pitch_score / max_score
    print(f"{folder_name}:\ntab accuracy: {tab_acc}\npitch accuracy: {pitch_acc}")
    with open(log_file_path, 'a') as file:
        file.write(f"{folder_name}:\ntab accuracy: {tab_acc}\npitch accuracy: {pitch_acc}" + '\n')
    return tab_acc, pitch_acc

if __name__ == '__main__':
    # for file in os.listdir(checkpoint_path):
    #     print(file)
    #     with open(log_file_path, 'a') as f:
    #         f.write(file + "\n")
    #     path = os.path.join(checkpoint_path, file)
    #     model.load_weights(path)
    #     overall_tab_acc = 0
    #     overall_pitch_acc = 0
    #     count = 0
    #     for label in chord_labels:
    #         count += 1
    #         tab_acc, pitch_acc = test_folder(label)
    #         overall_tab_acc += tab_acc
    #         overall_pitch_acc += pitch_acc
    #     overall_tab_acc /= count
    #     overall_pitch_acc /= count
    #     print(f"overall tab accuracy: {overall_tab_acc}\noverall pitch accuracy: {overall_pitch_acc}" + '\n')
    #     with open(log_file_path, 'a') as file:
    #         file.write(f"overall tab accuracy: {overall_tab_acc}\noverall pitch accuracy: {overall_pitch_acc}" + '\n')

    #
    overall_tab_acc = 0
    overall_pitch_acc = 0
    count = 0
    for label in chord_labels:
        count += 1
        tab_acc, pitch_acc = test_folder(label)
        overall_tab_acc += tab_acc
        overall_pitch_acc += pitch_acc
    overall_tab_acc /= count
    overall_pitch_acc /= count
    print(f"overall tab accuracy: {overall_tab_acc}\noverall pitch accuracy: {overall_pitch_acc}" + '\n')
    with open(log_file_path, 'a') as file:
        file.write(f"overall tab accuracy: {overall_tab_acc}\noverall pitch accuracy: {overall_pitch_acc}" + '\n')
