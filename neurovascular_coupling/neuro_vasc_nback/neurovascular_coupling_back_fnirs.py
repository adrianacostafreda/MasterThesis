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

# Set default directory
os.chdir("/Users/adriana/Documents/GitHub/thesis")
#os.chdir("H:\Dokumenter\GitHub\MasterThesis\.venv")
mne.set_log_level('error')
mne.set_config('MNE_BROWSER_BACKEND', 'matplotlib')

# Folder where to get the raw fNIRS files
clean_raw_fnirs_hc = "/Users/adriana/Documents/DTU/thesis/data_acquisition/data_fnirs/healthy_controls/"
clean_raw_fnirs_p = "/Users/adriana/Documents/DTU/thesis/data_acquisition/data_fnirs/patients/"

# Get directories of raw EEG files and set export directory for clean files
fnirs_dir_inprogress_hc = os.path.join(clean_raw_fnirs_hc)
fnirs_dir_inprogress_p = os.path.join(clean_raw_fnirs_p)

file_dirs_fnirs_hc, hc_eeg = read_files(fnirs_dir_inprogress_hc,'.snirf')
file_dirs_fnirs_p, p_eeg = read_files(fnirs_dir_inprogress_p,'.snirf')

epoch_duration = 1
delay_time = np.arange(0, 10, epoch_duration)

# healthy controls
subjects_hc = list()

# Loop over all files preprocess them with HemoData and save them at an numpy array.
# It save the data for each patient, for each event, for each channel:  patient X events X channels X time_samples.

for i in range(len(file_dirs_fnirs_hc)):

    # --------------Read Raw data -------------------
    # --------------Preprocess of fNIRS Raw data -------------------

    raw_haemo = HemoData(file_dirs_fnirs_hc[i], preprocessing=True, isPloting=False).getMneIoRaw()
    #raw_haemo.plot(block=True)

    epochs_fnirs_hc = characterization_fNIRS(raw_haemo, epoch_duration)
    #print("This is the list containing the epochs", len(epochs_fnirs_hc))

    hc_features_fnirs = list()

    # --------------Extract features-------------------------------- 
    for nback_hc in epochs_fnirs_hc:

        hc_coupling_list = list()

        for coupling_hc in range(len(nback_hc)):
            #print("This is each segment of the epoched data with delay", coupling[i])
            
            raw_data_hbo = nback_hc[coupling_hc].get_data(picks=["hbo"]) # epochs x channels x samples
            
            #print("This is the shape for raw_data_hbo", raw_data_hbo.shape)

            # mean of the concentration
            mean_per_epoch_per_channel = np.expand_dims(np.mean(raw_data_hbo, axis=-1), axis=-1) # epochs x channels X mean
            #print("This is the shape of mean_per_epoch_per_channel", mean_per_epoch_per_channel.shape)

            # std of the concentration HbO and HbR
            std_per_epoch_per_channel = np.expand_dims(np.std(raw_data_hbo, axis=-1), axis=-1) # epoch x channels X std
            #print("This is the shape of std_per_epoch_per_channel", std_per_epoch_per_channel.shape)

            # Concatenate along the last axis to get the shape (epochs, channels, 2)
            mean_std_per_epoch_per_channel = np.concatenate((mean_per_epoch_per_channel, std_per_epoch_per_channel), axis=-1)

            #print("This is the shape of mean_std_per_epoch_per_channel", mean_std_per_epoch_per_channel.shape)

            hc_coupling_list.append(mean_std_per_epoch_per_channel)
            print("This is the length of fnirs", len(hc_coupling_list))
        
        hc_features_fnirs.append(hc_coupling_list)
    
    print("This if the length of hc_features_fnirs", len(hc_features_fnirs))

    nback_hc = list()

    for n_back_hc in hc_features_fnirs:
        print("this is the length of nback", len(n_back_hc))

        hc_coupling_list_hbo = list()

        for coupling_hc in n_back_hc:

            coupling_hc = np.expand_dims(coupling_hc, axis=0)
            print("This is the shape of coupling", coupling_hc.shape)

            hc_coupling_list_hbo.append(coupling_hc)

        nback_hc.append(hc_coupling_list_hbo)

    subjects_hc.append(nback_hc)

print("This is the shape of subjects hc", len(subjects_hc))

# Dictionary to hold concatenated results for each delay and shape
concatenated_by_delay_hc = {}

# Iterate through each delay
for delay_index_hc in range(len(subjects_hc[0])):
    arrays_by_shape_hc = {}
    
    # Iterate through each subject
    for subject_hc in subjects_hc:
        for coupling_hc in subject_hc[delay_index_hc]:
            shape_hc = coupling_hc.shape
            
            if shape_hc not in arrays_by_shape_hc:
                arrays_by_shape_hc[shape_hc] = []
            arrays_by_shape_hc[shape_hc].append(coupling_hc)
    
    # Concatenate arrays with the same shape for the current delay
    concatenated_subject_hc = {}
    for shape_hc, arrays_hc in arrays_by_shape_hc.items():
        concatenated_array_hc = np.concatenate(arrays_hc, axis=0)
        concatenated_subject_hc[shape_hc] = concatenated_array_hc
    
    concatenated_by_delay_hc[delay_index_hc] = concatenated_subject_hc

delays_0back_hc = list()
delays_1back_hc = list()
delays_2back_hc = list()
delays_3back_hc = list()

# Example of how to access concatenated arrays for each delay and further separate them
for delay_index_hc, subject_hc in concatenated_by_delay_hc.items():
    print(f"Delay {delay_index_hc} concatenated arrays:")
    for shape_hc, array_hc in subject_hc.items():
        
        if delay_index_hc == 0:
            print(f"Shape 0back {shape_hc}: {array_hc.shape}")
            delays_0back_hc.append(array_hc)
        
        if delay_index_hc == 1:
            print(f"Shape 1back {shape_hc}: {array_hc.shape}")
            delays_1back_hc.append(array_hc)
        
        if delay_index_hc == 2:
            print(f"Shape 2back {shape_hc}: {array_hc.shape}")
            delays_2back_hc.append(array_hc)
        
        if delay_index_hc == 3:
            print(f"Shape 3back {shape_hc}: {array_hc.shape}")
            delays_3back_hc.append(array_hc)
                
print("This is the length of 0back", len(delays_0back_hc))
print("This is the length of 1back", len(delays_1back_hc))
print("This is the length of 2back", len(delays_2back_hc))
print("This is the length of 3back", len(delays_3back_hc))


# 0 back
hc_mean_0back = list()
for delay in delays_0back_hc:
    #print("This is the shape", delay.shape)
    hbo_concentration_hc = delay[:, :, : , 0] #(subj, freq band, epochs, channels)
    print("This is the shape of hbo concentration", hbo_concentration_hc.shape)
    mean_subj_hc = np.mean(hbo_concentration_hc, axis=0)
    #print("This is the shape of mean_p", mean_p.shape)
    mean_channels_hc = np.mean(mean_subj_hc, axis=-1)
    #print("This is the shape of mean_channels_p", mean_channels_p.shape)
    mean_epochs_hc = np.mean(mean_channels_hc, axis=0)
    #print("This is mean_epochs_hc", mean_epochs_hc)
    hc_mean_0back.append(mean_epochs_hc)

print("This is the length of hc_mean 0back ", hc_mean_0back)

# 1 back
hc_mean_1back = list()
for delay in delays_1back_hc:
    #print("This is the shape", delay.shape)
    hbo_concentration_hc = delay[:, :, : , 0] #(subj, freq band, epochs, channels)
    print("This is the shape of hbo concentration", hbo_concentration_hc.shape)
    mean_subj_hc = np.mean(hbo_concentration_hc, axis=0)
    #print("This is the shape of mean_p", mean_p.shape)
    mean_channels_hc = np.mean(mean_subj_hc, axis=-1)
    #print("This is the shape of mean_channels_p", mean_channels_p.shape)
    mean_epochs_hc = np.mean(mean_channels_hc, axis=0)
    #print("This is mean_epochs_hc", mean_epochs_hc)
    hc_mean_1back.append(mean_epochs_hc)

print("This is the length of hc_mean 1back", hc_mean_1back)

# 2 back
hc_mean_2back = list()
for delay in delays_2back_hc:
    #print("This is the shape", delay.shape)
    hbo_concentration_hc = delay[:, :, : , 0] #(subj, freq band, epochs, channels)
    print("This is the shape of hbo concentration", hbo_concentration_hc.shape)
    mean_subj_hc = np.mean(hbo_concentration_hc, axis=0)
    #print("This is the shape of mean_p", mean_p.shape)
    mean_channels_hc = np.mean(mean_subj_hc, axis=-1)
    #print("This is the shape of mean_channels_p", mean_channels_p.shape)
    mean_epochs_hc = np.mean(mean_channels_hc, axis=0)
    #print("This is mean_epochs_hc", mean_epochs_hc)
    hc_mean_2back.append(mean_epochs_hc)

print("This is the length of hc_mean 2back", hc_mean_2back)

# 3 back
hc_mean_3back = list()
for delay in delays_3back_hc:
    #print("This is the shape", delay.shape)
    hbo_concentration_hc = delay[:, :, : , 0] #(subj, freq band, epochs, channels)
    print("This is the shape of hbo concentration", hbo_concentration_hc.shape)
    mean_subj_hc = np.mean(hbo_concentration_hc, axis=0)
    #print("This is the shape of mean_p", mean_p.shape)
    mean_channels_hc = np.mean(mean_subj_hc, axis=-1)
    #print("This is the shape of mean_channels_p", mean_channels_p.shape)
    mean_epochs_hc = np.mean(mean_channels_hc, axis=0)
    #print("This is mean_epochs_hc", mean_epochs_hc)
    hc_mean_3back.append(mean_epochs_hc)

print("This is the length of hc_mean 3back", hc_mean_3back)

# patients
subjects_p = list()

# Loop over all files preprocess them with HemoData and save them at an numpy array.
# It save the data for each patient, for each event, for each channel:  patient X events X channels X time_samples.

for i in range(len(file_dirs_fnirs_p)):

    # --------------Read Raw data -------------------
    # --------------Preprocess of fNIRS Raw data -------------------

    raw_haemo = HemoData(file_dirs_fnirs_p[i], preprocessing=True, isPloting=False).getMneIoRaw()
    #raw_haemo.plot(block=True)

    epochs_fnirs_p = characterization_fNIRS(raw_haemo, epoch_duration)
    #print("This is the list containing the epochs", len(epochs_fnirs_p))

    p_features_fnirs = list()

    # --------------Extract features-------------------------------- 
    for nback_p in epochs_fnirs_p:

        p_coupling_list = list()

        for coupling_p in range(len(nback_p)):
            #print("This is each segment of the epoched data with delay", coupling[i])
            
            raw_data_hbo = nback_p[coupling_p].get_data(picks=["hbo"]) # epochs x channels x samples
            
            #print("This is the shape for raw_data_hbo", raw_data_hbo.shape)

            # mean of the concentration
            mean_per_epoch_per_channel = np.expand_dims(np.mean(raw_data_hbo, axis=-1), axis=-1) # epochs x channels X mean
            #print("This is the shape of mean_per_epoch_per_channel", mean_per_epoch_per_channel.shape)

            # std of the concentration HbO and HbR
            std_per_epoch_per_channel = np.expand_dims(np.std(raw_data_hbo, axis=-1), axis=-1) # epoch x channels X std
            #print("This is the shape of std_per_epoch_per_channel", std_per_epoch_per_channel.shape)

            # Concatenate along the last axis to get the shape (epochs, channels, 2)
            mean_std_per_epoch_per_channel = np.concatenate((mean_per_epoch_per_channel, std_per_epoch_per_channel), axis=-1)

            #print("This is the shape of mean_std_per_epoch_per_channel", mean_std_per_epoch_per_channel.shape)

            p_coupling_list.append(mean_std_per_epoch_per_channel)
            print("This is the length of fnirs", len(p_coupling_list))
        
        p_features_fnirs.append(p_coupling_list)
    
    print("This if the length of p_features_fnirs", len(p_features_fnirs))

    nback_p = list()

    for n_back_p in p_features_fnirs:
        print("this is the length of nback", len(n_back_p))

        p_coupling_list_hbo = list()

        for coupling_p in n_back_p:

            coupling_p = np.expand_dims(coupling_p, axis=0)
            print("This is the shape of coupling", coupling_p.shape)

            p_coupling_list_hbo.append(coupling_p)

        nback_p.append(p_coupling_list_hbo)

    subjects_p.append(nback_p)

print("This is the shape of subjects p", len(subjects_p))

# Dictionary to hold concatenated results for each delay and shape
concatenated_by_delay_p = {}

# Iterate through each delay
for delay_index_p in range(len(subjects_p[0])):
    arrays_by_shape_p = {}
    
    # Iterate through each subject
    for subject_p in subjects_p:
        for coupling_p in subject_p[delay_index_p]:
            shape_p = coupling_p.shape
            
            if shape_p not in arrays_by_shape_p:
                arrays_by_shape_p[shape_p] = []
            arrays_by_shape_p[shape_p].append(coupling_p)
    
    # Concatenate arrays with the same shape for the current delay
    concatenated_subject_p = {}
    for shape_p, arrays_p in arrays_by_shape_p.items():
        concatenated_array_p = np.concatenate(arrays_p, axis=0)
        concatenated_subject_p[shape_p] = concatenated_array_p
    
    concatenated_by_delay_p[delay_index_p] = concatenated_subject_p

delays_0back_p = list()
delays_1back_p = list()
delays_2back_p = list()
delays_3back_p = list()

# Example of how to access concatenated arrays for each delay and further separate them
for delay_index_p, subject_p in concatenated_by_delay_p.items():
    print(f"Delay {delay_index_p} concatenated arrays:")
    for shape_p, array_p in subject_p.items():
        
        if delay_index_p == 0:
            print(f"Shape 0back {shape_p}: {array_p.shape}")
            delays_0back_p.append(array_p)
        
        if delay_index_p == 1:
            print(f"Shape 1back {shape_p}: {array_p.shape}")
            delays_1back_p.append(array_p)
        
        if delay_index_p == 2:
            print(f"Shape 2back {shape_p}: {array_p.shape}")
            delays_2back_p.append(array_p)
        
        if delay_index_p == 3:
            print(f"Shape 3back {shape_p}: {array_p.shape}")
            delays_3back_p.append(array_p)
                
print("This is the length of 0back", len(delays_0back_p))
print("This is the length of 1back", len(delays_1back_p))
print("This is the length of 2back", len(delays_2back_p))
print("This is the length of 3back", len(delays_3back_p))


# 0 back
p_mean_0back = list()
for delay in delays_0back_p:
    print("This is the shape", delay.shape)
    hbo_concentration_p = delay[:, :, : , 0] #(subj, freq band, epochs, channels)
    print("This is the shape of hbo concentration", hbo_concentration_p.shape)
    mean_subj_p = np.mean(hbo_concentration_p, axis=0)
    #print("This is the shape of mean_p", mean_p.shape)
    mean_channels_p = np.mean(mean_subj_p, axis=-1)
    #print("This is the shape of mean_channels_p", mean_channels_p.shape)
    mean_epochs_p = np.mean(mean_channels_p, axis=0)
    #print("This is mean_epochs_p", mean_epochs_p)
    p_mean_0back.append(mean_epochs_p)

print("This is the length of p_mean 0back ", p_mean_0back)

# 1 back
p_mean_1back = list()
for delay in delays_1back_p:
    hbo_concentration_p = delay[:, :, : , 0] #(subj, freq band, epochs, channels)
    mean_subj_p = np.mean(hbo_concentration_p, axis=0)
    #print("This is the shape of mean_p", mean_p.shape)
    mean_channels_p = np.mean(mean_subj_p, axis=-1)
    mean_epochs_p = np.mean(mean_channels_p, axis=0)
    #print("This is mean_epochs_p", mean_epochs_p)
    p_mean_1back.append(mean_epochs_p)

print("This is the length of p_mean 1back", p_mean_1back)

# 2 back
p_mean_2back = list()
for delay in delays_2back_p:
    hbo_concentration_p = delay[:, :, : , 0] #(subj, freq band, epochs, channels)
    mean_subj_p = np.mean(hbo_concentration_p, axis=0)
    #print("This is the shape of mean_p", mean_p.shape)
    mean_channels_p = np.mean(mean_subj_p, axis=-1)
    mean_epochs_p = np.mean(mean_channels_p, axis=0)
    #print("This is mean_epochs_p", mean_epochs_p)
    p_mean_2back.append(mean_epochs_p)

print("This is the length of p_mean 2back", p_mean_2back)

# 3 back
p_mean_3back = list()
for delay in delays_3back_p:
    hbo_concentration_p = delay[:, :, : , 0] #(subj, freq band, epochs, channels)
    mean_subj_p = np.mean(hbo_concentration_p, axis=0)
    #print("This is the shape of mean_p", mean_p.shape)
    mean_channels_p = np.mean(mean_subj_p, axis=-1)
    mean_epochs_p = np.mean(mean_channels_p, axis=0)
    #print("This is mean_epochs_p", mean_epochs_p)
    p_mean_3back.append(mean_epochs_p)

print("This is the length of p_mean 3back", p_mean_3back)

fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# 0back subplot
axs[0, 0].plot(delay_time, p_mean_0back, label="patients")
axs[0, 0].plot(delay_time, hc_mean_0back, label="healthy")
axs[0, 0].set_xlabel("delay")
axs[0, 0].set_ylabel("relative theta power")
axs[0, 0].set_title("0back")
axs[0, 0].legend()

# 1back subplot
axs[0, 1].plot(delay_time, p_mean_1back, label="patients")
axs[0, 1].plot(delay_time, hc_mean_1back, label="healthy")
axs[0, 1].set_xlabel("delay")
axs[0, 1].set_ylabel("relative theta power")
axs[0, 1].set_title("1back")
axs[0, 1].legend()

# 2back subplot
axs[1, 0].plot(delay_time, p_mean_2back, label="patients")
axs[1, 0].plot(delay_time, hc_mean_2back, label="healthy")
axs[1, 0].set_xlabel("delay")
axs[1, 0].set_ylabel("relative theta power")
axs[1, 0].set_title("2back")
axs[1, 0].legend()

# 3back subplot
axs[1, 1].plot(delay_time, p_mean_3back, label="patients")
axs[1, 1].plot(delay_time, hc_mean_3back, label="healthy")
axs[1, 1].set_xlabel("delay")
axs[1, 1].set_ylabel("relative theta power")
axs[1, 1].set_title("3back")
axs[1, 1].legend()

plt.tight_layout()
plt.show()


