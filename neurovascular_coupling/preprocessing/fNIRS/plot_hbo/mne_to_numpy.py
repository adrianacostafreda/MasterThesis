import numpy as np
import mne

from DataPath import DataPath
from Hemo import HemoData
from sklearn.preprocessing import StandardScaler

import mne_nirs
from itertools import compress
import matplotlib.pyplot as plt
from mne.decoding import UnsupervisedSpatialFilter
from scipy.spatial.distance import pdist, squareform
from mne._fiff.pick import pick_channels, pick_info, pick_types
from sklearn.decomposition import PCA

import os
from queue import LifoQueue


def invertDic(dic: dict):
        '''
        Helping function to invert the keys and the values of a dictionary.
        '''
        inv_dic = {v: k for k, v in dic.items()}
        return inv_dic

def characterization_trigger_data(raw):
    
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
            print(annot.duration[idx])
            duration_trigger = annot.duration[idx] + 60 # Durations of the test are 60 seconds
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

    #----------------------------------------------------------------------------------

    #----------------------------------------------------------------------------------

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
    
    raw = raw.set_annotations(new_annotations)
    
    #----------------------------------------------------------------------------------
    events, _ = mne.events_from_annotations(raw) # Create events from existing annotations
    
    print("These are the events", events)
    
    event_id = {"4": 1, "5": 2, "6": 3, "7": 4}

    # Using this event array, we take continuous data and create epochs ranging from -0.5 seconds before to 60 seconds after each event for 5,6,7 and 48 seconds for 4 
    # In other words, an epoch comprises data from -0.5 to 60 seconds around 5, 6, 7 events. For event 4, the epoch will comprise data from -0.5 to 48 seconds. 
    # We will consider Fz, Cz, Pz channels corresponding to the mid frontal line
    tmin=0
    tmax=40
    epochs = mne.Epochs(
        raw,
        events, 
        event_id,
        tmin,
        tmax,
        baseline=(0,0),
        preload=True
    )
    
    return epochs

path_hc = "/Users/adriana/Documents/DTU/thesis/data_acquisition/data_fnirs/healthy_controls/"
#path = "H:\\Dokumenter\\data_acquisition\\data_fnirs\\patients\\baseline\\snirf_files\\"
path_p = "/Users/adriana/Documents/DTU/thesis/data_acquisition/data_fnirs/patients/"

datapath_hc = DataPath(path_hc, recursive=False)
datapath_p = DataPath(path_p, recursive=False)

print(len(datapath_hc.getDataPaths()))
print(len(datapath_p.getDataPaths()))

# Loop over all files preprocess them with HemoData and save them at an numpy array.
# It save the data for each patient, for each event, for each channel:  patient X events X channels X time_samples.
for id, file in enumerate(datapath_hc.getDataPaths()):
        raw_haemo = HemoData(file, preprocessing=True, isPloting=False).getMneIoRaw()
        #raw_haemo.plot(block=True)

        epochs = characterization_trigger_data(raw_haemo)

        # --------------Epoched Data--------------------------------
        # Each epoched object contains 1s epoched data corresponding to each data segment

        raw_data_hbo = epochs.get_data(picks=["hbo"]) # epochs x channels x samples
        raw_data_hbr = epochs.get_data(picks=["hbr"]) # epochs x channels x samples

        # Initialize StandardScaler
        ss = StandardScaler()

        # Apply StandardScaler independently to each slice along the first dimension (axis=0)
        for i in range(raw_data_hbo.shape[0]):
            raw_data_hbo[i] = ss.fit_transform(raw_data_hbo[i])
            raw_data_hbr[i] = ss.fit_transform(raw_data_hbr[i])

        print("This is the shape of raw_data_hbo", raw_data_hbo.shape)

        # We will expand each task 15 s which corresponds to 153
        t_interval = 153

        # Simple version subjects x events x channels x samples.
        if id == 0:
            data_hbo_hc = np.expand_dims(raw_data_hbo[:, :, :],axis=0)
            data_hbr_hc = np.expand_dims(raw_data_hbr[:, :, :],axis=0)
        else:
            data_hbo_hc = np.concatenate((data_hbo_hc, np.expand_dims(raw_data_hbo[:, :, :],axis=0)),axis=0)
            data_hbr_hc = np.concatenate((data_hbr_hc, np.expand_dims(raw_data_hbr[:, :, :],axis=0)),axis=0)


# Shape of data (n_subjects, epochs, channels, samples)
print("This is the shape of HBO data", data_hbo_hc.shape)
print("This is the shape of HBR data", data_hbr_hc.shape)

for id, file in enumerate(datapath_p.getDataPaths()):
        raw_haemo = HemoData(file, preprocessing=True, isPloting=False).getMneIoRaw()
        #raw_haemo.plot(block=True)

        epochs = characterization_trigger_data(raw_haemo)

        # --------------Epoched Data--------------------------------
        # Each epoched object contains 1s epoched data corresponding to each data segment

        raw_data_hbo = epochs.get_data(picks=["hbo"]) # epochs x channels x samples
        raw_data_hbr = epochs.get_data(picks=["hbr"]) # epochs x channels x samples

        # Initialize StandardScaler
        ss = StandardScaler()

        # Apply StandardScaler independently to each slice along the first dimension (axis=0)
        for i in range(raw_data_hbo.shape[0]):
            raw_data_hbo[i] = ss.fit_transform(raw_data_hbo[i])
            raw_data_hbr[i] = ss.fit_transform(raw_data_hbr[i])

        print("This is the shape of raw_data_hbo", raw_data_hbo.shape)

        # We will expand each task 15 s which corresponds to 153
        t_interval = 153

        # Simple version subjects x events x channels x samples.
        if id == 0:
            data_hbo_p = np.expand_dims(raw_data_hbo[:, :, :],axis=0)
            data_hbr_p = np.expand_dims(raw_data_hbr[:, :, :],axis=0)
        else:
            data_hbo_p = np.concatenate((data_hbo_p, np.expand_dims(raw_data_hbo[:, :, :],axis=0)),axis=0)
            data_hbr_p = np.concatenate((data_hbr_p, np.expand_dims(raw_data_hbr[:, :, :],axis=0)),axis=0)

# Calculate the mean across the 0 axis
mean_data_hc = np.mean(data_hbo_hc, axis=0)  # Shape: (4, 8, 612)
mean_data_p = np.mean(data_hbo_p, axis=0)

# Number of events and subplots per event
num_events = mean_data_hc.shape[0]  # 4
num_subplots = mean_data_hc.shape[1]  # 8

num_samples_hc = mean_data_hc.shape[2]  # 612
num_samples_hc = mean_data_p[2]

# Create plots
for event_idx in range(num_events):
    fig, axes = plt.subplots(4, 4, figsize=(15, 15))
    fig.suptitle(f'Nback {event_idx}', fontsize=16)
    
    for subplot_idx in range(num_subplots):
        row, col = divmod(subplot_idx, 4)
        ax = axes[row, col]
        ax.plot(mean_data_hc[event_idx, subplot_idx, :], label="healthy controls")
        ax.plot(mean_data_p[event_idx, subplot_idx, :], label ="patients")
        ax.set_title(f'Channel {subplot_idx + 1}')
        ax.set_xlabel('Samples')
        ax.set_ylabel('Amplitude')
        ax.legend()
    
    # Hide any unused subplots
    for idx in range(num_subplots, 16):
        row, col = divmod(idx, 4)
        axes[row, col].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.subplots_adjust(hspace=0.5, wspace=0.5)  # Adjust the spacing between subplots
    plt.show()


