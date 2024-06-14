import numpy as np
import mne

from neurovascular_coupling.DataPath import DataPath
from neurovascular_coupling.Hemo import HemoData
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
    
    raw_new=raw.set_annotations(new_annotations)
    #raw_new.plot(block=True)
    #plt.show()
    
    #----------------------------------------------------------------------------------

    raw_0back = raw_new.copy().crop(tmin=trigger_data['4']['begin'][0], tmax=trigger_data['4']['end'][0])
    raw_1back = raw_new.copy().crop(tmin=trigger_data['5']['begin'][0], tmax=trigger_data['5']['end'][0])
    raw_2back = raw_new.copy().crop(tmin=trigger_data['6']['begin'][0], tmax=trigger_data['6']['end'][0])
    raw_3back = raw_new.copy().crop(tmin=trigger_data['7']['begin'][0], tmax=trigger_data['7']['end'][0])
    
    return (raw_0back, raw_1back, raw_2back, raw_3back)

def epochs_from_raw(raw, raw_0back, raw_1back, raw_2back, raw_3back, epoch_duration):
    """
    Returns a mne.Epochs object created from the annotated mne.io.Raw object.

    """
    # Create epochs of 1 s for all raw segments 

    events_0back = mne.make_fixed_length_events(raw_0back, start = 0, stop = 48, duration = epoch_duration)
    events_1back = mne.make_fixed_length_events(raw_1back, start = 0, stop = 60, duration = epoch_duration)
    events_2back = mne.make_fixed_length_events(raw_2back, start = 0, stop = 60, duration = epoch_duration)
    events_3back = mne.make_fixed_length_events(raw_3back, start = 0, stop = 60, duration = epoch_duration)

    interval = (0,0)
    epochs_0back = mne.Epochs(raw_0back, events_0back, baseline=interval,preload=True)
    epochs_1back = mne.Epochs(raw_1back, events_1back, baseline=interval,preload=True)
    epochs_2back = mne.Epochs(raw_2back, events_2back, baseline=interval,preload=True)
    epochs_3back = mne.Epochs(raw_3back, events_3back, baseline=interval,preload=True)
    
    print(epochs_0back)
    print(epochs_1back)
    print(epochs_2back)
    print(epochs_3back)

    return (epochs_0back, epochs_1back, epochs_2back, epochs_3back)

#path = "/Users/adriana/Documents/DTU/thesis/data_acquisition/data_fnirs/healthy_controls/"
path = "H:\\Dokumenter\\data_acquisition\\data_fnirs\\patients\\baseline\\snirf_files\\"

datapath = DataPath(path, recursive=False)
print(len(datapath.getDataPaths()))

epoch_duration = 0.5

# Loop over all files preprocess them with HemoData and save them at an numpy array.
# It save the data for each patient, for each event, for each channel:  patient X events X channels X time_samples.
for id,file in enumerate(datapath.getDataPaths()):
        raw_haemo = HemoData(file, preprocessing=True, isPloting=False).getMneIoRaw()
        #raw_haemo.plot(block=True)

        raw_characterization = characterization_trigger_data(raw_haemo)

        raw_0back = raw_characterization[0]
        raw_1back = raw_characterization[1]
        raw_2back = raw_characterization[2]
        raw_3back = raw_characterization[3]

        # --------------Epoched Data--------------------------------
        # Each epoched object contains 1s epoched data corresponding to each data segment

        epochs = epochs_from_raw(raw_haemo, raw_0back, raw_1back, raw_2back, raw_3back, epoch_duration)

        raw_data_hbo_0back = epochs[0].get_data(picks=["hbo"]) # epochs x channels x samples
        raw_data_hbr_0back = epochs[0].get_data(picks=["hbr"]) # epochs x channels x samples

        raw_data_hbo_1back = epochs[1].get_data(picks=["hbo"]) # epochs x channels x samples
        raw_data_hbr_1back = epochs[1].get_data(picks=["hbr"]) # epochs x channels x samples

        raw_data_hbo_2back = epochs[2].get_data(picks=["hbo"]) # epochs x channels x samples
        raw_data_hbr_2back = epochs[2].get_data(picks=["hbr"]) # epochs x channels x samples

        raw_data_hbo_3back = epochs[3].get_data(picks=["hbo"]) # epochs x channels x samples
        raw_data_hbr_3back = epochs[3].get_data(picks=["hbr"]) # epochs x channels x samples

        # Initialize StandardScaler
        ss = StandardScaler()

        # Apply StandardScaler independently to each slice along the first dimension (axis=0)
        for i in range(raw_data_hbo_0back.shape[0]):
            raw_data_hbo_0back[i] = ss.fit_transform(raw_data_hbo_0back[i])
            raw_data_hbr_0back[i] = ss.fit_transform(raw_data_hbr_0back[i])
        
        for i in range(raw_data_hbo_1back.shape[0]):
            raw_data_hbo_1back[i] = ss.fit_transform(raw_data_hbo_1back[i])
            raw_data_hbr_1back[i] = ss.fit_transform(raw_data_hbr_1back[i])

        for i in range(raw_data_hbo_2back.shape[0]):
            raw_data_hbo_2back[i] = ss.fit_transform(raw_data_hbo_2back[i])
            raw_data_hbr_2back[i] = ss.fit_transform(raw_data_hbr_2back[i])
        
        for i in range(raw_data_hbo_3back.shape[0]):
            raw_data_hbo_3back[i] = ss.fit_transform(raw_data_hbo_3back[i])
            raw_data_hbr_3back[i] = ss.fit_transform(raw_data_hbr_3back[i])


        print("This is the shape of raw_data_hbo_0back", raw_data_hbr_0back.shape)
        print("This is the shape of raw_data_hbo_1back", raw_data_hbr_1back.shape)
        print("This is the shape of raw_data_hbo_2back", raw_data_hbr_2back.shape)
        print("This is the shape of raw_data_hbo_3back", raw_data_hbr_3back.shape)


        # We will expand each task 15 s which corresponds to 153
        t_interval = 153

        # Simple version subjects x events x channels x samples.
        if id == 0:
            data_hbo_0back = np.expand_dims(raw_data_hbo_0back[:, :, :],axis=0)
            data_hbr_0back = np.expand_dims(raw_data_hbr_0back[:, :, :],axis=0)
        else:
            data_hbo_0back = np.concatenate((data_hbo_0back, np.expand_dims(raw_data_hbo_0back[:, :, :],axis=0)),axis=0)
            data_hbr_0back = np.concatenate((data_hbr_0back, np.expand_dims(raw_data_hbr_0back[:, :, :],axis=0)),axis=0)

        # Simple version subjects x events x channels x samples.
        if id == 0:
            data_hbo_1back = np.expand_dims(raw_data_hbo_1back[:, :, :],axis=0)
            data_hbr_1back = np.expand_dims(raw_data_hbr_1back[:, :, :],axis=0)
        else:
            data_hbo_1back = np.concatenate((data_hbo_1back, np.expand_dims(raw_data_hbo_1back[:, :, :],axis=0)),axis=0)
            data_hbr_1back = np.concatenate((data_hbr_1back, np.expand_dims(raw_data_hbr_1back[:, :, :],axis=0)),axis=0)

        # Simple version subjects x events x channels x samples.
        if id == 0:
            data_hbo_2back = np.expand_dims(raw_data_hbo_2back[:, :, :],axis=0)
            data_hbr_2back = np.expand_dims(raw_data_hbr_2back[:, :, :],axis=0)
        else:
            data_hbo_2back = np.concatenate((data_hbo_2back, np.expand_dims(raw_data_hbo_2back[:, :, :],axis=0)),axis=0)
            data_hbr_2back = np.concatenate((data_hbr_2back, np.expand_dims(raw_data_hbr_2back[:, :, :],axis=0)),axis=0)

        # Simple version subjects x events x channels x samples.
        if id == 0:
            data_hbo_3back = np.expand_dims(raw_data_hbo_3back[:, :, :],axis=0)
            data_hbr_3back = np.expand_dims(raw_data_hbr_3back[:, :, :],axis=0)
        else:
            data_hbo_3back = np.concatenate((data_hbo_3back, np.expand_dims(raw_data_hbo_3back[:, :, :],axis=0)),axis=0)
            data_hbr_3back = np.concatenate((data_hbr_3back, np.expand_dims(raw_data_hbr_3back[:, :, :],axis=0)),axis=0)

# Shape of data (n_subjects, epochs, channels, samples)
print("This is the shape of 0back HBO data", data_hbo_0back.shape)
print("This is the shape of 0back HBR data", data_hbr_0back.shape)

print("This is the shape of 1back HBO data", data_hbo_1back.shape)
print("This is the shape of 1back HBR data", data_hbr_1back.shape)

print("This is the shape of 2back HBO data", data_hbo_2back.shape)
print("This is the shape of 2back HBR data", data_hbr_2back.shape)

print("This is the shape of 3back HBO data", data_hbo_3back.shape)
print("This is the shape of 3back HBR data", data_hbr_3back.shape)

# Save all the data paths

# Save data path

clean_path_0back = "H:\\Dokumenter\\data_processing\\Results_fNIRS\\patients\\0_back\\"
clean_path_1back = "H:\\Dokumenter\\data_processing\\Results_fNIRS\\patients\\1_back\\"
clean_path_2back = "H:\\Dokumenter\\data_processing\\Results_fNIRS\\patients\\2_back\\"
clean_path_3back = "H:\\Dokumenter\\data_processing\\Results_fNIRS\\patients\\3_back\\"

os.makedirs(clean_path_0back, exist_ok=True)
os.makedirs(clean_path_1back, exist_ok=True)
os.makedirs(clean_path_2back, exist_ok=True)
os.makedirs(clean_path_3back, exist_ok=True)

# 0back HBO & HBR
np.save(clean_path_0back + "hemo_0back", data_hbo_0back) 
np.save(clean_path_0back + "dehemo_0back", data_hbr_0back)

# 1back HBO & HBR
np.save(clean_path_1back + "hemo_1back", data_hbo_1back) 
np.save(clean_path_1back + "dehemo_1back", data_hbr_1back) 

# 2back HBO & HBR
np.save(clean_path_2back + "hemo_2back", data_hbo_2back) 
np.save(clean_path_2back + "dehemo_2back", data_hbr_2back) 

# 3back HBO & HBR
np.save(clean_path_3back + "hemo_3back", data_hbo_3back)
np.save(clean_path_3back + "dehemo_3back", data_hbr_3back) 

