import pandas as pd
import numpy as np
import os
import basic.process_group_psd_data as process_psd
import basic.arrange_files as arrange
import mne
import antropy

from IPython.display import display

current_directory = "H:\Dokumenter\GitHub\MasterThesis"

os.chdir("H:\Dokumenter\GitHub\MasterThesis\.venv")
mne.set_log_level('error')

epochs_data_folder = "H:\\Dokumenter\\data_acquisition\\data_eeg\\clean\\healthy_controls\\n_back\\"

psd_reg_folder = "H:\\Dokumenter\\data_processing\\Results\\healthy_controls\\n_back\\Relative PSD\\regions\\"
psd_ch_folder =  "H:\\Dokumenter\\data_processing\\Results\\healthy_controls\\n_back\\Relative PSD\\channels\\"

condition_legend = ['healthy_controls','patients']

exp_folder = ["0_back", "1_back", "2_back", "3_back"]
condition_codes = ['0-Back','1-Back', '2-Back', '3-Back']
condition_codes_comparisons = [['0-Back','1-Back', '2-Back', '3-Back']]

print('N-back')
[df_psd_reg , df_psd_ch , epochs] = process_psd.read_group_psd_data(psd_reg_folder, psd_ch_folder,
                                                    exp_folder, non_responders=None, data_folder = "H:\\Dokumenter\\data_acquisition\\data_eeg\\clean\\healthy_controls\\n_back\\")

# Keep only the rows of the theta power
df_psd_ch = df_psd_ch[df_psd_ch['Frequency band'] == "Theta"]

df_psd_ch['Subject'] = df_psd_ch['Subject'].str.replace("_0back","").str.replace("_1back","").str.replace("_2back","").str.replace("_3back","")
df_psd_reg['Subject'] = df_psd_reg['Subject'].str.replace("_0back","").str.replace("_1back","").str.replace("_2back","").str.replace("_3back","")

# List of columns to select
columns_to_select = ["Subject", "AFp1", "AFp2", "AFF1h", "AFF2h", "FFC1h", "FFC2h", "FCC1h", "FCC2h", "Condition", "Frequency band"]

# Select the desired columns
df_psd_ch = df_psd_ch.loc[:, columns_to_select]

print(df_psd_ch)

# create separate data frames for each of the tasks
df_psd_ch_back0 = df_psd_ch[df_psd_ch['Condition'] == "0"]
df_psd_ch_back1 = df_psd_ch[df_psd_ch['Condition'] == "1"]
df_psd_ch_back2 = df_psd_ch[df_psd_ch['Condition'] == "2"]
df_psd_ch_back3 = df_psd_ch[df_psd_ch['Condition'] == "3"]


# ----------------------------------------------0 Back------------------------------------------------------ 
subjects = df_psd_ch_back0['Subject'].unique()
channels = df_psd_ch_back0.columns[1:-2]  # Exclude the first column (Subject) and last two columns (Frequency band and Condition)

# Create an empty NumPy array
features_0back = np.zeros((len(subjects), len(channels), 1))

# Iterate through the DataFrame and fill in the NumPy array
for i, subject in enumerate(subjects):
    # Filter rows for the current subject
    subject_rows = df_psd_ch_back0[df_psd_ch_back0['Subject'] == subject]
    # Extract power data for the subject
    subject_power = subject_rows[channels].values
    # Reshape power data to (number of channels, 1)
    subject_power = subject_power.reshape(len(channels), 1)
    # Assign power data to the corresponding position in data_array
    features_0back[i, :, :] = subject_power

# ----------------------------------------------1 Back------------------------------------------------------ 
subjects = df_psd_ch_back1['Subject'].unique()
channels = df_psd_ch_back1.columns[1:-2]  # Exclude the first column (Subject) and last two columns (Frequency band and Condition)

# Create an empty NumPy array
features_1back = np.zeros((len(subjects), len(channels), 1))

# Iterate through the DataFrame and fill in the NumPy array
for i, subject in enumerate(subjects):
    # Filter rows for the current subject
    subject_rows = df_psd_ch_back1[df_psd_ch_back1['Subject'] == subject]
    # Extract power data for the subject
    subject_power = subject_rows[channels].values
    # Reshape power data to (number of channels, 1)
    subject_power = subject_power.reshape(len(channels), 1)
    # Assign power data to the corresponding position in data_array
    features_1back[i, :, :] = subject_power

# ----------------------------------------------2 Back------------------------------------------------------ 
subjects = df_psd_ch_back2['Subject'].unique()
channels = df_psd_ch_back2.columns[1:-2]  # Exclude the first column (Subject) and last two columns (Frequency band and Condition)

# Create an empty NumPy array
features_2back = np.zeros((len(subjects), len(channels), 1))

# Iterate through the DataFrame and fill in the NumPy array
for i, subject in enumerate(subjects):
    # Filter rows for the current subject
    subject_rows = df_psd_ch_back2[df_psd_ch_back2['Subject'] == subject]
    # Extract power data for the subject
    subject_power = subject_rows[channels].values
    # Reshape power data to (number of channels, 1)
    subject_power = subject_power.reshape(len(channels), 1)
    # Assign power data to the corresponding position in data_array
    features_2back[i, :, :] = subject_power

# ----------------------------------------------3 Back------------------------------------------------------ 
subjects = df_psd_ch_back3['Subject'].unique()
channels = df_psd_ch_back3.columns[1:-2]  # Exclude the first column (Subject) and last two columns (Frequency band and Condition)

# Create an empty NumPy array
features_3back = np.zeros((len(subjects), len(channels), 1))

# Iterate through the DataFrame and fill in the NumPy array
for i, subject in enumerate(subjects):
    # Filter rows for the current subject
    subject_rows = df_psd_ch_back3[df_psd_ch_back3['Subject'] == subject]
    # Extract power data for the subject
    subject_power = subject_rows[channels].values
    # Reshape power data to (number of channels, 1)
    subject_power = subject_power.reshape(len(channels), 1)
    # Assign power data to the corresponding position in data_array
    features_3back[i, :, :] = subject_power

# Save all the numpy arrays

#clean_path = "/Users/adriana/Documents/DTU/thesis/data_processing/Results_fNIRS/"
clean_path = "H:\\Dokumenter\\data_processing\\Results_EEG\\features\\"

np.save(clean_path + "features_0Back", features_0back)
np.save(clean_path + "features_1Back", features_1back)
np.save(clean_path + "features_2Back", features_2back)
np.save(clean_path + "features_3Back", features_3back)
