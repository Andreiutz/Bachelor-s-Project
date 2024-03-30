import numpy as np

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