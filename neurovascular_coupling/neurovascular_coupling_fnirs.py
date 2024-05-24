import mne
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
from scipy.integrate import simps
from scipy.signal import welch
import yasa
from mnelab.io.xdf import read_raw_xdf

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

epoch_duration = 1

# Set default directory
os.chdir("/Users/adriana/Documents/GitHub/thesis")
#os.chdir("H:\Dokumenter\GitHub\MasterThesis\.venv")
mne.set_log_level('error')
mne.set_config('MNE_BROWSER_BACKEND', 'matplotlib')

delay_time = np.arange(0, 10, epoch_duration)

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

    # make a 10 s delay
    for i in delay:
        epochs = list()
        events_0back = mne.make_fixed_length_events(raw_0back, start = 0 + i, stop= 58, duration = epoch_duration)
        events_1back = mne.make_fixed_length_events(raw_1back, start = 0 + i, stop= 70, duration = epoch_duration)
        events_2back = mne.make_fixed_length_events(raw_2back, start = 0 + i, stop= 70, duration = epoch_duration)
        events_3back = mne.make_fixed_length_events(raw_3back, start = 0 + i, stop= 60, duration = epoch_duration)
        
        interval = (0,0)
        epochs_0back = mne.Epochs(raw_0back, events_0back, baseline = interval, preload = True)
        epochs_1back = mne.Epochs(raw_1back, events_1back, baseline = interval, preload = True)
        epochs_2back = mne.Epochs(raw_2back, events_2back, baseline = interval, preload = True)
        epochs_3back = mne.Epochs(raw_3back, events_3back, baseline = interval, preload = True)

        epochs.append(epochs_0back)
        epochs.append(epochs_1back)
        epochs.append(epochs_2back)
        epochs.append(epochs_3back)

        fnirs_epochs_coupling.append(epochs)
    
    #print("This is the length of fnirs_epochs", fnirs_epochs_coupling)
    #print("This is the length of fnirs_epochs", len(fnirs_epochs_coupling))

    return fnirs_epochs_coupling

#raw_fnirs_folder = "H:\\Dokumenter\\data_acquisition\\data_fnirs\\healthy_controls\\baseline\\snirf_files\\first_trial\\"
clean_raw_fnirs_hc = "/Users/adriana/Documents/DTU/thesis/data_acquisition/data_fnirs/healthy_controls/"
clean_raw_fnirs_p = "/Users/adriana/Documents/DTU/thesis/data_acquisition/data_fnirs/patients/"

# Get directories of raw EEG files and set export directory for clean files
fnirs_dir_inprogress_hc = os.path.join(clean_raw_fnirs_hc)
eeg_dir_inprogress_p = os.path.join(clean_raw_fnirs_p)

file_dirs_fnirs_hc, subjects_hc_fnirs = read_files(fnirs_dir_inprogress_hc,'.snirf')
file_dirs_fnirs_p, subjects_p_fnirs = read_files(eeg_dir_inprogress_p,'.snirf')

# Healthy controls
subjects_hc = list()

# Loop over all files preprocess them with HemoData and save them at an numpy array.
# It save the data for each patient, for each event, for each channel:  patient X events X channels X time_samples.

for i in range(len(file_dirs_fnirs_hc)):

    # --------------Read Raw data -------------------
    # --------------Preprocess of fNIRS Raw data -------------------

    raw_haemo = HemoData(file_dirs_fnirs_hc[i], preprocessing=True, isPloting=False).getMneIoRaw()
    #raw_haemo.plot(block=True)

    epochs_fnirs_hc = characterization_fNIRS(raw_haemo, epoch_duration)
    #print("This is the list containing the epochs", epochs_fnirs_hc)

    hc_features_fnirs = list()

    # --------------Extract features-------------------------------- 
    for coupling in epochs_fnirs_hc:
        concatenated_data = list()

        for i in range(len(coupling)):
            #print("This is each segment of the epoched data with delay", coupling[i])
            
            raw_data_hbo = coupling[i].get_data(picks=["hbo"]) # epochs x channels x samples
            
            #print("This is the shape for raw_data_hbo", raw_data_hbo.shape)
            
            # Append the data array to the list
            concatenated_data.append(raw_data_hbo)

        concatenated_hbo = np.concatenate(concatenated_data, axis = 0)
        #print("This is the shape of concatenated_hbo", concatenated_hbo.shape)

        # mean of the concentration
        mean_per_epoch_per_channel = np.expand_dims(np.mean(concatenated_hbo, axis=-1), axis=-1) # epochs x channels X mean
        #print("This is the shape of mean_per_epoch_per_channel", mean_per_epoch_per_channel.shape)

        # std of the concentration HbO and HbR
        std_per_epoch_per_channel = np.expand_dims(np.std(concatenated_hbo, axis=-1), axis=-1) # epoch x channels X std
        #print("This is the shape of std_per_epoch_per_channel", std_per_epoch_per_channel.shape)

        # Concatenate along the last axis to get the shape (epochs, channels, 2)
        mean_std_per_epoch_per_channel = np.concatenate((mean_per_epoch_per_channel, std_per_epoch_per_channel), axis=-1)

        #print("This is the shape of mean_std_per_epoch_per_channel", mean_std_per_epoch_per_channel.shape)

        hc_features_fnirs.append(mean_std_per_epoch_per_channel)
    
    coupling_fnirs_hc = list()
    for coupling_fnirs in hc_features_fnirs:
        
        coupling_fnirs = np.expand_dims(coupling_fnirs, axis=0)
        #print("This is the shape of coupling", coupling.shape)
        coupling_fnirs_hc.append(coupling_fnirs)

    subjects_hc.append(coupling_fnirs_hc)

delays_hc = [list() for _ in range(10)]

for i in subjects_hc:
    for j in range(10):
        delays_hc[j].append(i[j])

delays_hc = [np.concatenate(delay, axis=0) for delay in delays_hc]

#for delay in delays_hc:
#    print("This is the shape of delay", delay.shape)

hc_mean = list()
for delay in delays_hc:
    #print("This is the shape", delay.shape)
    hbo_mean_hc = delay[:, :, : , 0] #(subj, freq band, epochs, channels)
    hbo_mean_subj_hc = np.mean(hbo_mean_hc, axis=0) # we compute the mean across all subjects 
    #print("This is the shape of mean_hc", hbo_mean_subj_hc.shape)
    hbo_mean_channels_hc = np.mean(hbo_mean_subj_hc, axis=-1)
    #print("This is the shape of mean_channels_hc", hbo_mean_channels_hc.shape)
    mean_hbo_epochs_hc = np.mean(hbo_mean_channels_hc, axis=0)
    #print("This is mean_epochs_hc", mean_hbo_epochs_hc)
    hc_mean.append(mean_hbo_epochs_hc)

#print("This is the length of hc_mean", len(hc_mean))

# PATIENTS

subjects_p = list()
# Loop over all files preprocess them with HemoData and save them at an numpy array.
# It save the data for each patient, for each event, for each channel:  patient X events X channels X time_samples.

for i in range(len(file_dirs_fnirs_p)):

    # --------------Read Raw data -------------------
    # --------------Preprocess of fNIRS Raw data -------------------

    raw_haemo = HemoData(file_dirs_fnirs_p[i], preprocessing=True, isPloting=False).getMneIoRaw()
    #raw_haemo.plot(block=True)

    epochs_fnirs_p = characterization_fNIRS(raw_haemo, epoch_duration)
    #print("This is the list containing the epochs", epochs_fnirs_hc)

    p_features_fnirs = list()

    # --------------Extract features-------------------------------- 
    for coupling_p in epochs_fnirs_p:
        concatenated_data_p = list()

        for i in range(len(coupling_p)):
            #print("This is each segment of the epoched data with delay", coupling[i])
            
            raw_data_hbo_p = coupling_p[i].get_data(picks=["hbo"]) # epochs x channels x samples
            
            #print("This is the shape for raw_data_hbo", raw_data_hbo.shape)
            
            # Append the data array to the list
            concatenated_data_p.append(raw_data_hbo_p)

        concatenated_hbo_p = np.concatenate(concatenated_data_p, axis = 0)
        #print("This is the shape of concatenated_hbo", concatenated_hbo_p.shape)

        # mean of the concentration
        p_mean_per_epoch_per_channel = np.expand_dims(np.mean(concatenated_hbo_p, axis=-1), axis=-1) # epochs x channels X mean
        #print("This is the shape of mean_per_epoch_per_channel", mean_per_epoch_per_channel.shape)

        # std of the concentration HbO and HbR
        p_std_per_epoch_per_channel = np.expand_dims(np.std(concatenated_hbo_p, axis=-1), axis=-1) # epoch x channels X std
        #print("This is the shape of std_per_epoch_per_channel", std_per_epoch_per_channel.shape)

        # Concatenate along the last axis to get the shape (epochs, channels, 2)
        p_mean_std_per_epoch_per_channel = np.concatenate((p_mean_per_epoch_per_channel, p_std_per_epoch_per_channel), axis=-1)

        p_features_fnirs.append(p_mean_std_per_epoch_per_channel)
    
    coupling_fnirs_p_list = list()
    for coupling_fnirs_p in p_features_fnirs:
        
        coupling_fnirs_p = np.expand_dims(coupling_fnirs_p, axis=0)
        coupling_fnirs_p_list.append(coupling_fnirs_p)

    subjects_p.append(coupling_fnirs_p_list)

delays_p = [list() for _ in range(10)]

for i in subjects_p:
    for j in range(10):
        delays_p[j].append(i[j])

delays_p = [np.concatenate(delay_p, axis=0) for delay_p in delays_p]

#for delay_p in delays_p:
    #print("This is the shape of delay", delay_p.shape)

p_mean = list()
for delay_p in delays_p:
    #print("This is the shape", delay.shape)
    hbo_mean_p = delay_p[:, :,: , 0] #(subj, freq band, epochs, channels)
    hbo_mean_subj_p = np.mean(hbo_mean_p, axis=0) # we compute the mean across all subjects 
    #print("This is the shape of mean_p", hbo_mean_subj_p.shape)
    hbo_mean_channels_p = np.mean(hbo_mean_subj_p, axis=-1)
    #print("This is the shape of mean_channels_p", hbo_mean_channels_p.shape)
    mean_hbo_epochs_p = np.mean(hbo_mean_channels_p, axis=0)
    #print("This is mean_epochs_p", mean_hbo_epochs_p)
    p_mean.append(mean_hbo_epochs_p)

#print("This is the length of hc_mean", len(p_mean))

plt.plot(delay_time, p_mean, label="patients")
plt.plot(delay_time, hc_mean, label = "healthy")
plt.xlabel("delay")
plt.ylabel("hbo concentration")
plt.legend()
plt.show()