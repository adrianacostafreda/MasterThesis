import mne
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
from scipy.integrate import simps
from scipy.signal import welch
import yasa

import mne_nirs
from itertools import compress
from mne.decoding import UnsupervisedSpatialFilter
from scipy.spatial.distance import pdist, squareform
from mne._fiff.pick import pick_channels, pick_info, pick_types
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import os
from queue import LifoQueue

from arrange_files import read_files
from Hemo import HemoData
from FeatureExtraction import FeatureExtraction


def empty_directory(directory_path):
    # Check if the directory exists
    if os.path.exists(directory_path):
        # Iterate over all the files and directories in the specified directory
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            try:
                # Remove file or directory
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # Remove file or link
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Remove directory and its contents
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
    else:
        # Create the directory if it doesn't exist
        os.makedirs(directory_path)
        print(f'Directory {directory_path} created.')

# fNIRS

def invertDic(dic: dict):
        '''
        Helping function to invert the keys and the values of a dictionary.
        '''
        inv_dic = {v: k for k, v in dic.items()}
        return inv_dic

def characterization_fNIRS(raw, epoch_duration):
    
    # Dictionary to store trigger data
    trigger_data = {
        '4': {'begin': [], 'end': [], 'duration': []},
        '5': {'begin': [], 'end': [], 'duration': []},
        '6': {'begin': [], 'end': [], 'duration': []},
        '7': {'begin': [], 'end': [], 'duration': []}
    }
    annot = raw.annotations
        
    # Triggers 5, 6 & 7 --> 1-Back Test, 2-Back Test, 3-Back Test
    # Iterate over annotations 
    for idx, desc in enumerate(annot.description):
        if desc in trigger_data:
            begin_trigger = annot.onset[idx] 
            duration_trigger = annot.duration[idx] + 50 # duration of 1, 2, 3 back is 60 s
            end_trigger = begin_trigger + duration_trigger

            # in case the file ends before
            if end_trigger >= raw.times[-1]:
                end_trigger = raw.times[-1]
            else:
                end_trigger = end_trigger
                
            #store trigger data in the dictionary
            trigger_data[desc]["begin"].append(begin_trigger)
            trigger_data[desc]["end"].append(end_trigger)
            trigger_data[desc]["duration"].append(duration_trigger)
    
    #----------------------------------------------------------------------------------

    # Determine the start of trigger 4
    # to set the beginning on Trigger 4, we compute the beginning of Trigger 5 and we subtract the relax
    # period (15s), and the test time (48s) & instructions (8s) of 1-Back Test
    # we add 10 seconds to start the segmentation 10 seconds after the trigger

    begin_trigger4 = trigger_data["5"]["begin"][0] - 15 - 48 - 8
    trigger_data["4"]["begin"].append(begin_trigger4)
    trigger_data["4"]["duration"].append(48) # Duration of 0-Back Test is 48 s

    end_time = trigger_data["4"]["begin"][0] + 48
    trigger_data["4"]["end"].append(end_time)

    # Set new annotations for the current file
    onsets = []
    durations = []
    descriptions = []

    # Accumulate trigger data into lists
    for description, data in trigger_data.items():
        for onset, duration in zip(data["begin"], data["duration"]):
            onsets.append(onset)
            durations.append(duration)
            descriptions.append(description)

    # Create new annotations
    new_annotations = mne.Annotations(
    onsets,
    durations,
    descriptions,
    ch_names=None  # Set to None since annotations don't have associated channel names
    )
    
    raw_new = raw.set_annotations(new_annotations)
    
    #----------------------------------------------------------------------------------

    #print("fnirs 0back", trigger_data['4']['end'][0] + 10 - trigger_data['4']['begin'][0])
    #print("fnirs 1back", trigger_data['5']['end'][0] + 10 - trigger_data['5']['begin'][0])
    #print("fnirs 2back", trigger_data['6']['end'][0] + 10 - trigger_data['6']['begin'][0])
    #print("fnirs 3back", trigger_data['7']['end'][0] - trigger_data['7']['begin'][0])

    raw_0back = raw_new.copy().crop(tmin=trigger_data['4']['begin'][0], tmax=trigger_data['4']['end'][0] + 11)
    raw_1back = raw_new.copy().crop(tmin=trigger_data['5']['begin'][0], tmax=trigger_data['5']['end'][0] + 11)
    raw_2back = raw_new.copy().crop(tmin=trigger_data['6']['begin'][0], tmax=trigger_data['6']['end'][0] + 11)
    raw_3back = raw_new.copy().crop(tmin=trigger_data['7']['begin'][0], tmax=trigger_data['7']['end'][0] + 1)

    # Epoch data

    fnirs_epochs_coupling = list()

    #delay = np.arange(epoch_duration, 10 + epoch_duration, epoch_duration)
    delay = np.arange(0 + epoch_duration, 10 + epoch_duration, epoch_duration)
    
    epochs_0back_list = list()
    epochs_1back_list = list()
    epochs_2back_list = list()
    epochs_3back_list = list()

    for i in delay:
        events_0back = mne.make_fixed_length_events(raw_0back, start = 0 , stop = 58 + epoch_duration - i, duration = epoch_duration)
        
        interval = (0,0)
        epochs_0back = mne.Epochs(raw_0back, events_0back, baseline = interval, preload = True)
        epochs_0back_list.append(epochs_0back)
    #print("This is the length of epochs_0back", len(epochs_0back_list))

    for i in delay:
        events_1back = mne.make_fixed_length_events(raw_1back, start = 0 , stop = 70 + epoch_duration - i, duration = epoch_duration)

        interval = (0,0)
        epochs_1back = mne.Epochs(raw_1back, events_1back, baseline = interval, preload = True)
        epochs_1back_list.append(epochs_1back)
    #print("This is the length of epochs_1back", len(epochs_1back_list))

    for i in delay:
        events_2back = mne.make_fixed_length_events(raw_2back, start = 0 , stop = 70 + epoch_duration - i, duration = epoch_duration)

        interval = (0,0)
        epochs_2back = mne.Epochs(raw_2back, events_2back, baseline = interval, preload = True)
        epochs_2back_list.append(epochs_2back)
    #print("This is the length of epochs_2back", len(epochs_2back_list))

    for i in delay:
        events_3back = mne.make_fixed_length_events(raw_3back, start = 0 , stop = 60 + epoch_duration - i, duration = epoch_duration)

        interval = (0,0)
        epochs_3back = mne.Epochs(raw_3back, events_3back, baseline = interval, preload = True)
        epochs_3back_list.append(epochs_3back)
    #print("This is the length of epochs_3back", len(epochs_3back_list))

    fnirs_epochs_coupling.append(epochs_0back_list)
    fnirs_epochs_coupling.append(epochs_1back_list)
    fnirs_epochs_coupling.append(epochs_2back_list)
    fnirs_epochs_coupling.append(epochs_3back_list)

    #print("This is the length eeg_epochs", len(eeg_epochs_coupling))
    #print("This is the length eeg_epochs", eeg_epochs_coupling)

    return fnirs_epochs_coupling

def process_files(file_dirs, epoch_duration):
    subjects = []
    for file in file_dirs:
        raw_haemo = HemoData(file, preprocessing=True, isPloting=False).getMneIoRaw()
        epochs_fnirs = characterization_fNIRS(raw_haemo, epoch_duration)

        features_hbo_nback = []

        for nback in epochs_fnirs:
            coupling_list = []
            for segment in nback:
                raw_data_hbo = segment.get_data(picks=["hbo"])
                print("This is the shape of raw_data_hbo",raw_data_hbo.shape)
                mean = np.expand_dims(np.mean(raw_data_hbo, axis=-1), axis=-1)
                std = np.expand_dims(np.std(raw_data_hbo, axis=-1), axis=-1)
                mean_std = np.concatenate((mean, std), axis=-1)
                #print("This is the shape of mean_std", mean_std.shape)
                coupling_list.append(mean_std)
            features_hbo_nback.append(coupling_list)

        subject_data = []

        for n_back in features_hbo_nback:
            coupling_list_bp = [np.expand_dims(coupling, axis=0) for coupling in n_back]
            subject_data.append(coupling_list_bp)

        subjects.append(subject_data)

    return subjects

def concatenate_by_delay(subjects):
    concatenated_by_delay = {}
    for delay_index in range(len(subjects[0])):
        arrays_by_shape = {}
        for subject in subjects:
            for coupling in subject[delay_index]:
                shape = coupling.shape
                if shape not in arrays_by_shape:
                    arrays_by_shape[shape] = []
                arrays_by_shape[shape].append(coupling)
        concatenated_subject = {shape: np.concatenate(arrays, axis=0) for shape, arrays in arrays_by_shape.items()}
        concatenated_by_delay[delay_index] = concatenated_subject

    return concatenated_by_delay

def extract_delays(concatenated_by_delay):
    delays = [[] for _ in range(len(concatenated_by_delay))]
    for delay_index, subject in concatenated_by_delay.items():
        for shape, array in subject.items():
            delays[delay_index].append(array)
    return delays

def calculate_means(delays, label):
    means_back = []
    for delay in delays:
        all_means = []
        
        for sub_delay in delay:
            #print("This is the shape of delay", sub_delay.shape)
            mean_hbo = sub_delay[:, :, :, 0]
            mean_subj = np.mean(mean_hbo, axis=0)
            mean_channels = np.mean(mean_subj, axis=-1)
            mean_epochs = np.mean(mean_channels, axis=0)
            #print("This is the shape of mean epochs", mean_epochs.shape)
            all_means.append(mean_epochs) 

        means_back.append(all_means)

    print(f"This is the length of {label} mean: {len(means_back)}")

    return means_back

def plot_means(delay_time, hc_means, p_means):
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    titles = ["0back", "1back", "2back", "3back"]
    for i, (hc_mean, p_mean, title) in enumerate(zip(hc_means, p_means, titles)):
        ax = axs[i//2, i%2]
        ax.plot(delay_time, p_mean, label="patients")
        ax.plot(delay_time, hc_mean, label="healthy")
        ax.set_xlabel("delay")
        ax.set_ylabel("hbo concentration")
        ax.set_title(title)
        ax.legend()
    plt.tight_layout()
    plt.show()

# Main Execution
os.chdir("/Users/adriana/Documents/GitHub/thesis")
mne.set_log_level('error')
mne.set_config('MNE_BROWSER_BACKEND', 'matplotlib')

clean_raw_fnirs_hc = "/Users/adriana/Documents/DTU/thesis/data_acquisition/data_fnirs/healthy_controls/"
clean_raw_fnirs_p = "/Users/adriana/Documents/DTU/thesis/data_acquisition/data_fnirs/patients/"

file_dirs_fnirs_hc, _ = read_files(clean_raw_fnirs_hc, '.snirf')
file_dirs_fnirs_p, _ = read_files(clean_raw_fnirs_p, '.snirf')

epoch_duration = 1
delay_time = np.arange(0, 10, epoch_duration)

subjects_hc = process_files(file_dirs_fnirs_hc, epoch_duration)
subjects_p = process_files(file_dirs_fnirs_p, epoch_duration)

concatenated_by_delay_hc = concatenate_by_delay(subjects_hc)
concatenated_by_delay_p = concatenate_by_delay(subjects_p)

delays_hc = extract_delays(concatenated_by_delay_hc)
delays_p = extract_delays(concatenated_by_delay_p)

hc_means = calculate_means(delays_hc, "hc_mean")
p_means = calculate_means(delays_p, "p_mean")

plot_means(delay_time, hc_means, p_means)

