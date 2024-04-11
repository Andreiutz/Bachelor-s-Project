import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
def tab_to_pitch(tab):
    result = []
    open_pitch_values = [40, 45, 50, 55, 59, 64]
    for index, string_vector in enumerate(tab):
        fret_index = np.argmax(string_vector)
        #fret_index = 0 means that the string is muted
        if fret_index > 0:
            midi_value = fret_index + open_pitch_values[index] - 1
            if midi_value not in result:
                result.append(midi_value)
    return result


# aligns equal pitches
def process_pitches(pred_pitch_list, gt_pitch_list):
    i = 0
    j = 0
    while i < len(pred_pitch_list) and j < len(gt_pitch_list):
        if pred_pitch_list[i] > gt_pitch_list[j]:
            pred_pitch_list = pred_pitch_list[:i] + [0] + pred_pitch_list[i:]
        elif pred_pitch_list[i] < gt_pitch_list[j]:
            gt_pitch_list = gt_pitch_list[:j] + [0] + gt_pitch_list[j:]

        i += 1
        j += 1
    ppllen = len(pred_pitch_list)
    gpllen = len(gt_pitch_list)
    pred_pitch_list += [0] * (gpllen-j)
    gt_pitch_list += [0] * (ppllen-i)
    return pred_pitch_list, gt_pitch_list

def compute_pitch_list_from_archive(y_pred_archive, y_gt_archive):
    pred = []
    gt = []
    for i in range(len(y_gt_archive)):
        pred_pitches = tab_to_pitch(y_pred_archive[i])
        gt_pitches = tab_to_pitch(y_gt_archive[i])
        pred_pitches.sort()
        gt_pitches.sort()

        pred_pitches, gt_pitches = process_pitches(pred_pitches, gt_pitches)
        if (len(pred_pitches) != len(gt_pitches)):
            raise "Something went wrong..."

        pred += pred_pitches
        gt += gt_pitches

    return pred, gt

def notes_and_positions(tab):
    result = {
        "notes" : [],
        "frets" : []
    }
    midi_pitches = [40, 45, 50, 55, 59, 64]
    for i, string in enumerate(tab):
        fret_index = np.argmax(string)
        #there is actually a note
        if fret_index > 0:
            result["frets"].append(fret_index-1)
            result["notes"].append(midi_pitches[i] + fret_index - 1)
        else:
            result["frets"].append(-1)
            result["notes"].append(0)
    return result

def plot_confusion_matrix(pred, gt, title, labels=None, size=(20,15), xlabel='', ylabel=''):
    if labels is None:
        labels = []
    table = pd.DataFrame(confusion_matrix(gt, pred))
    plt.figure(figsize=size)
    ax = sns.heatmap(table, annot=True, fmt='d', cmap='viridis',  annot_kws={"size": 15})
    ax.set_title(title, fontsize=20)
    if len(labels) != 0:
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels, rotation=0)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.show()