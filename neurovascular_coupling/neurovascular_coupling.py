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
raw_fnirs_folder_hc = "/Users/adriana/Documents/DTU/thesis/data_acquisition/data_fnirs/healthy_controls/"

# Get directories of raw EEG files and set export directory for clean files
fnirs_dir_inprogress_hc = os.path.join(raw_fnirs_folder_hc)

file_dirs_fnirs_hc, subject_names_fnirs_hc = read_files(fnirs_dir_inprogress_hc,'.snirf')

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
        print("This is the shape of concatenated_hbo", concatenated_hbo.shape)

        # mean of the concentration
        mean_per_epoch_per_channel = np.expand_dims(np.mean(concatenated_hbo, axis=-1), axis=-1) # epochs x channels X mean
        #print("This is the shape of mean_per_epoch_per_channel", mean_per_epoch_per_channel.shape)

        # std of the concentration HbO and HbR
        std_per_epoch_per_channel = np.expand_dims(np.std(concatenated_hbo, axis=-1), axis=-1) # epoch x channels X std
        #print("This is the shape of std_per_epoch_per_channel", std_per_epoch_per_channel.shape)

        # Concatenate along the last axis to get the shape (epochs, channels, 2)
        mean_std_per_epoch_per_channel = np.concatenate((mean_per_epoch_per_channel, std_per_epoch_per_channel), axis=-1)

        print("This is the shape of mean_std_per_epoch_per_channel", mean_std_per_epoch_per_channel.shape)

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

for delay in delays_hc:
    print("This is the shape of delay", delay.shape)

    #print("This is the length of hc_data_fnirs", len(hc_data_fnirs))

    ##subjects_hc.append(hc_data_fnirs)

#print("This is the length of subjects_hc", len(subjects_hc))

#for subject in subjects_hc:
#    for coupling in subject:
        #print("This is the shape of coupling", coupling.shape)

"""



# Define a function to extract the numeric part from the filename
def extract_numeric_part(filename):
    return int(filename.split('.')[0].split('coupling')[1])

# fNIRS extract features 

for subject in subject_names_fnirs:

    delay_count = 0

    fnirs_samples_subject = os.path.join(fnirs_samples, subject)

    # Get a list of all files in the folder
    file_list_fnirs = os.listdir(fnirs_samples_subject)

    # Sort the list using a custom key for numeric sorting
    file_list_fnirs.sort(key=extract_numeric_part)

    # Join the folder path with each file name to get the full path
    full_paths_fnirs = [os.path.join(fnirs_samples_subject, file_name) for file_name in file_list_fnirs]

    # Extract HBO concentration
    features_hbo_list = list()

    for full_path in full_paths_fnirs:
        #print(full_path)

        feature_extractor = FeatureExtraction(full_path)
        features_hbo = feature_extractor.getFeatures() # numpy array
        #print("Printing features HBO numpy shape", features_hbo.shape)

        features_hbo_list.append(features_hbo)
             
    for feature in features_hbo_list:
        #print("This is the feature shape", feature.shape)

        path_fnirs_features = '{}/{}/'.format(fnirs_features, subject)
        os.makedirs(path_fnirs_features, exist_ok=True)

        np.save(path_fnirs_features + "coupling" + str(delay_count), feature)

        delay_count = delay_count + 1 


#Correlation analysis for one subject

import scipy
from scipy.stats import pearsonr, spearmanr

# I have to read the numpy files in fnirs_features and EEG_bandpower

#eeg

print("This is the subject EEG we are evaluating", subject_names_eeg[1])

subject_names_eeg_corr = subject_names_eeg[1]

eeg_bp_subject = os.path.join(eeg_bandpower, subject_names_eeg_corr)

# Get a list of all files in the folder
file_list_eeg_features = os.listdir(eeg_bp_subject)

file_list_eeg_features.sort(key=extract_numeric_part)

# Filter the list to only include NumPy files
numpy_files_eeg_features = [file_eeg for file_eeg in file_list_eeg_features if file_eeg.endswith('.npy')]

# Join the folder path with each file name to get the full path
full_paths_eeg_features = [os.path.join(eeg_bp_subject, file_name_eeg) for file_name_eeg in numpy_files_eeg_features]

corr_features_eeg = list()
for full_path_eeg in full_paths_eeg_features:
    #print(full_path_eeg)

    features_eeg = np.load(full_path_eeg)
    #print("These is the size of the features", features_eeg.shape)

    theta = features_eeg[1 , : , :] #(freq band, epochs, channels)

    mean_channels_theta = np.mean(theta, axis=-1)  # shape: (epochs,)

    corr_features_eeg.append(mean_channels_theta)

#fnirs

print("This is the subject fNIRS we are evaluating", subject_names_fnirs[1])

subject_names_fnirs_corr = subject_names_fnirs[1]

fnirs_hbo_subject = os.path.join(fnirs_features, subject_names_fnirs_corr)

# Get a list of all files in the folder
file_list_fnirs_features = os.listdir(fnirs_hbo_subject)

file_list_fnirs_features.sort(key=extract_numeric_part)

# Filter the list to only include NumPy files
numpy_files_fnirs_features = [file_fnirs for file_fnirs in file_list_fnirs_features if file_fnirs.endswith('.npy')]

# Join the folder path with each file name to get the full path
full_paths_fnirs_features = [os.path.join(fnirs_hbo_subject, file_name_fnirs) for file_name_fnirs in numpy_files_fnirs_features]

corr_features_fnirs = list()

for full_path_fnirs in full_paths_fnirs_features:
    #print(full_path_fnirs)

    features_fnirs = np.load(full_path_fnirs)
    #print("These is the size of the features", features_fnirs.shape)

    mean_hbo = features_fnirs[:, 0:4, 0] # (epochs, channels, feature)

    mean_channels_hbo = np.mean(mean_hbo, axis=-1)  # shape: (epochs,)

    corr_features_fnirs.append(mean_channels_hbo)

#print("This is the length of corr_features_eeg", len(order_corr_features_fnirs))

correlation_list = list()
p_value_list = list()

for fnirs_hbo, eeg_theta in zip(corr_features_fnirs, corr_features_eeg):
    #print("This is the shape for the corr FNIRS features which includes the epochs", fnirs_hbo.shape)
    #print("This is the shape for the corr EEG features which includes the epochs", eeg_theta.shape)
    
    # Calculate correlation
    correlation, p_value = pearsonr(fnirs_hbo, eeg_theta)
    #print("Correlation coefficient:", correlation)
    #print("P-value:", p_value)
    correlation_list.append(correlation)
    p_value_list.append(p_value)

#print("This is the length of correlation", len(correlation_list))
#print("This is the length of pvalue", len(p_value_list))

delay = np.arange(0, 10, epoch_duration)

# Crear una figura con subplots
fig, ax = plt.subplots(2, 1, figsize=(10, 8))

# Primer subplot: pearson r correlation
ax[0].plot(delay, correlation_list)
ax[0].set_title("Pearson r Correlation")
ax[0].set_xlabel("Delay")
ax[0].set_ylabel("Correlation")

# Segundo subplot: p-values correlation
ax[1].plot(delay, p_value_list)
ax[1].set_title("P-values Correlation")
ax[1].set_xlabel("Delay")
ax[1].set_ylabel("P-value")

# Ajustar el layout para evitar superposición
plt.tight_layout()

# Mostrar la figura
plt.show()
"""