import mne
import numpy as np
import matplotlib.pyplot as plt
import os

from arrange_files import read_files

# Set default directory
#os.chdir("H:\Dokumenter\GitHub\MasterThesis\.venv")
os.chdir("/Users/adriana/Documents/GitHub/thesis")
mne.set_log_level('error')

# Folder where to get the raw EEG files
#clean_raw_eeg_folder = "H:\\Dokumenter\\data_acquisition\\data_eeg\\clean_eeg\\healthy_controls\\first_trial\\"
# raw_fnirs_folder = "H:\\Dokumenter\\data_acquisition\\data_fnirs\\healthy_controls\\baseline\\snirf_files\\first_trial\\"

clean_raw_eeg_folder = "/Users/adriana/Documents/DTU/thesis/data_acquisition/clean_eeg/healthy_controls/"
raw_fnirs_folder = "/Users/adriana/Documents/DTU/thesis/data_acquisition/data_fnirs/healthy_controls/"

# Get directories of raw EEG files and set export directory for clean files
eeg_dir_inprogress = os.path.join(clean_raw_eeg_folder)
# Get directories of raw fNIRS files and set export directory for clean files
fnirs_dir_inprogress = os.path.join(raw_fnirs_folder)

file_dirs_eeg, subject_names_eeg = read_files(eeg_dir_inprogress, '.fif')
file_dirs_fnirs, subject_names_fnirs = read_files(fnirs_dir_inprogress, '.snirf')

#eeg_bandpower = "H:\\Dokumenter\\data_processing\\neurovascular_coupling\\EEG_bandpower\\"
#fnirs_features = "H:\\Dokumenter\\data_processing\\neurovascular_coupling\\fnirs_features\\"

eeg_bandpower = "/Users/adriana/Documents/DTU/thesis/data_processing/neurovascular_coupling/EEG_bandpower/"
fnirs_features = "/Users/adriana/Documents/DTU/thesis/data_processing/neurovascular_coupling/fnirs_features/"

# Define a function to extract the numeric part from the filename
def extract_numeric_part(filename):
    return int(filename.split('.')[0].split('coupling')[1])

"""
Correlation analysis for one subject

"""

import scipy
from scipy.stats import pearsonr, spearmanr

# I have to read the numpy files in fnirs_features and EEG_bandpower

epoch_duration = 0.8

"""
eeg
"""

corr_features_eeg_list = list()

for eeg_subject in range(len(subject_names_eeg)):

    subject_names_eeg_corr = subject_names_eeg[eeg_subject]

    eeg_bp_subject = os.path.join(eeg_bandpower, subject_names_eeg_corr)

    # Get a list of all files in the folder
    file_list_eeg_features = os.listdir(eeg_bp_subject)

    # Sort the list using a custom key for numeric sorting
    file_list_eeg_features.sort(key=extract_numeric_part)

    # Filter the list to only include NumPy files
    numpy_files_eeg_features = [file_eeg for file_eeg in file_list_eeg_features if file_eeg.endswith('.npy')]

    # Join the folder path with each file name to get the full path
    full_paths_eeg_features = [os.path.join(eeg_bp_subject, file_name_eeg) for file_name_eeg in numpy_files_eeg_features]

    corr_features_eeg = list()

    for full_path_eeg in full_paths_eeg_features:
        #print(full_path_eeg)

        features_eeg = np.load(full_path_eeg)
        

        theta = features_eeg[1 , : , :] #(freq band, epochs, channels)
        mean_channels_theta = np.mean(theta, axis=-1)  # shape: (epochs,)

        corr_features_eeg.append(mean_channels_theta)

    corr_features_eeg_list.append(corr_features_eeg)


"""
fnirs
"""

corr_features_fnirs_list = list()

for subject_fnirs in range(len(subject_names_fnirs)):

    subject_names_fnirs_corr = subject_names_fnirs[subject_fnirs]

    fnirs_hbo_subject = os.path.join(fnirs_features, subject_names_fnirs_corr)

    # Get a list of all files in the folder
    file_list_fnirs_features = os.listdir(fnirs_hbo_subject)

    # Sort the list using a custom key for numeric sorting
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

    corr_features_fnirs_list.append(corr_features_fnirs)

correlation_list_subj = list()
p_value_list_subj = list()

#print("This is the length of corr_features_eeg, corr_features_fnirs", len(corr_features_eeg_list), len(corr_features_fnirs_list))

for subject_eeg, subject_fnirs in zip(corr_features_eeg_list, corr_features_fnirs_list):

    correlation_list = list()
    p_value_list = list()

    for eeg_theta, fnirs_hbo in zip(subject_eeg, subject_fnirs):
        
        print("This is the shape for the corr FNIRS features which includes the epochs", fnirs_hbo.shape)
        print("This is the shape for the corr EEG features which includes the epochs", eeg_theta.shape)
        
        # Calculate correlation
        correlation, p_value = pearsonr(fnirs_hbo, eeg_theta)
        #print("Correlation coefficient:", correlation)
        #print("P-value:", p_value)
        correlation_list.append(correlation)
        p_value_list.append(p_value)
    
    correlation_list_subj.append(correlation_list)
    p_value_list_subj.append(p_value_list)

print("This is the length of correlation", correlation_list_subj)
print("This is the length of pvalue", p_value_list_subj)

delay = np.arange(0, 10, epoch_duration)

# Color cycle for different lines
colors = plt.cm.viridis(np.linspace(0, 1, len(correlation_list_subj)))

# Create a figure with two subplots (one row, two columns)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

for corr_value, p_value, color in zip(correlation_list_subj, p_value_list_subj, colors):

    # First subplot: Pearson r correlation
    ax1.plot(delay, corr_value, color=color)
    ax1.set_title("Pearson r Correlation")
    ax1.set_xlabel("Delay")
    ax1.set_ylabel("Correlation")

    # Second subplot: p-values correlation
    ax2.plot(delay, p_value, color=color)
    ax2.set_title("P-values Correlation")
    ax2.set_xlabel("Delay")
    ax2.set_ylabel("P-value")

# Add legends to both subplots
#ax1.legend()
#ax2.legend()

# Adjust layout for better spacing
plt.tight_layout()

# Show the plots
plt.show()
