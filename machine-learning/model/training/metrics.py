import numpy as np
import keras

#todo refactor all to accept another input format

def tab_to_pitch(tab):
    result = np.zeros(44)
    open_pitch_values = [40, 45, 50, 55, 59, 64]
    for index, string_vector in enumerate(tab):
        fret_index = np.argmax(string_vector)
        #fret_index = 0 means that the string is muted
        if fret_index > 0:
            result[fret_index + open_pitch_values[index] - 41] = 1
    return result


def tab_without_closed_class(tab):
    tab_arr = np.zeros((6, 20))
    for string_num in range(len(tab)):
        fret_vector = tab[string_num]
        fret_class = np.argmax(fret_vector, -1)
        # 0 means that the string is closed
        if fret_class > 0:
            fret_num = fret_class - 1
            tab_arr[string_num][fret_num] = 1
    return tab_arr


def pitch_precision(predictions, ground_truths):
    pitch_predictions = np.array(
        [tab_to_pitch(pred) for pred in predictions]
    )
    pitch_ground_truths = np.array(
        [tab_to_pitch(gt) for gt in ground_truths]
    )
    total_correct_pitches = np.sum(np.multiply(pitch_predictions, pitch_ground_truths).flatten())
    total_predicted_pitches = np.sum(pitch_predictions.flatten())
    return (1.0 * total_correct_pitches) / total_predicted_pitches


def pitch_recall(predictions, ground_truths):
    pitch_predictions = np.array(
        [tab_to_pitch(pred) for pred in predictions]
    )
    pitch_ground_truths = np.array(
        [tab_to_pitch(gt) for gt in ground_truths]
    )
    total_correct_pitches = np.sum(np.multiply(pitch_predictions, pitch_ground_truths).flatten())
    total_ground_truth_pitches = np.sum(pitch_ground_truths.flatten())
    return (1.0 * total_correct_pitches) / total_ground_truth_pitches


def pitch_f_score(predictions, ground_truths):
    precision = pitch_precision(predictions, ground_truths)
    recall = pitch_recall(predictions, ground_truths)
    return (2 * precision * recall) / (precision + recall)


def tab_precision(predictions, ground_truths):
    # get rid of "closed" class, as we only want to count positives
    tab_predictions = np.array(
        [tab_without_closed_class(pred) for pred in predictions]
    )
    tab_ground_truths = np.array(
        [tab_without_closed_class(gt) for gt in ground_truths]
    )

    numerator = np.sum(np.multiply(tab_predictions, tab_ground_truths).flatten())
    denominator = np.sum(tab_predictions.flatten())
    return (1.0 * numerator) / denominator


def tab_recall(pred, gt):
    # get rid of "closed" class, as we only want to count positives
    tab_pred = np.array([tab_without_closed_class(p) for p in pred])
    tab_gt = np.array([tab_without_closed_class(g) for g in gt])
    numerator = np.sum(np.multiply(tab_pred, tab_gt).flatten())
    denominator = np.sum(tab_gt.flatten())
    return (1.0 * numerator) / denominator


def tab_f_measure(pred, gt):
    p = tab_precision(pred, gt)
    r = tab_recall(pred, gt)
    f = (2 * p * r) / (p + r)
    return f


def tab_disamb(pred, gt):
    tp = tab_precision(pred, gt)
    pp = pitch_precision(pred, gt)
    return tp / pp