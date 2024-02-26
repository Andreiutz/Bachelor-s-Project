import numpy as np
import librosa
from typing import List

# given a set of samples which were determined to be part of same note, return the strum
def get_predicted_strum_of_batch(y, start_index, end_index):
    result = []
    for string in y:
        summed_arr = np.average(string[start_index : end_index], axis=0)
        argmax_index = int(np.argmax(summed_arr))
        if argmax_index == 0:
            argmax_value = summed_arr[argmax_index]
            aux_array = np.delete(summed_arr, argmax_index)
            new_argmax_index = int(np.argmax(aux_array))
            new_argmax_value = aux_array[new_argmax_index]
            if abs(argmax_value - new_argmax_value) < 0.05: # todo look into a better threshold
                argmax_index = new_argmax_index + 1 #we deleted the first value, so the index should shift up with one value
        result.append(argmax_index-1)               #to be as before deleting
    return result


# returns the strum of a single sample (1 / 43) of a second
def get_predicted_strum_of_sample(y):
    result = []
    for string in y:
        result.append(int(np.argmax(string)) - 1)
    return result

#returns indexes of the frames when there is a change is the sound input
def get_onset_frames(y, sr):
    unfiltered_frames =  librosa.onset.onset_detect(
        y=y,
        sr=sr,
        wait=2.5,
        pre_avg=1,
        post_avg=1,
        pre_max=1,
        post_max=1,
        backtrack=True,
        normalize=True
    )
    unfiltered_frames = np.insert(unfiltered_frames, 0, 0) # insert first frame before all
    unfiltered_frames = np.insert(unfiltered_frames, len(unfiltered_frames), int((len(y)-1) / 512 + 1))
    filtered_frames = []
    for i in range(len(unfiltered_frames)-1):
        start_frame = unfiltered_frames[i]
        end_frame = unfiltered_frames[i+1]
        start_audio_sample = start_frame * 512
        end_audio_sample = end_frame * 512
        if max(y[start_audio_sample : end_audio_sample]) > 0.3 and end_frame - start_frame > 3: # end_frame - start_frame > 3 -> 0.06 seconds aprox
            filtered_frames.append([start_frame, end_frame])
    return filtered_frames
