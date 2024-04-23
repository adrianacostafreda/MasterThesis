# Import packages
import os, mne
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import neurokit2 as nk

from IPython.display import display

# Set default directory
os.chdir("H:\Dokumenter\GitHub\MasterThesis\.venv")
mne.set_log_level('error')

# Import functions
import basic.arrange_files as arrange

# Folder where to get the clean epochs files
clean_folder = "H:\\Dokumenter\\data_acquisition\\data_eeg\\clean\\"

# Folder where to save the results
results_foldername = "H:\\Dokumenter\\data_processing\\entropy_complexity\\"

# Sub-folder for the experiment (i.e. timepoint or group) and its acronym
exp_folder = 'n_back'
exp_condition = '0_back'

# Get directories of clean EEG files and set export directory
dir_inprogress = os.path.join(clean_folder,exp_folder, exp_condition)
file_dirs, subject_names = arrange.read_files(dir_inprogress,'_clean-epo.fif')

lzc_args = dict(symbolize='median')
mse_args = dict(method="IMSEn", scale=14, dimension=3, tolerance='sd')
#best MSE methods --> MMSEn, IMSEn (link: https://neuropsychology.github.io/NeuroKit/functions/complexity.html#entropy-multiscale)

# Loop through all the subjects' directories (EEG files directories)
df = pd.DataFrame(index=subject_names)
for i in range(len(file_dirs)):
    # Read the clean data from the disk
    print('\n{} in progress:'.format(subject_names[i]))
    epochs = mne.read_epochs(fname='{}/{}_clean-epo.fif'.format(dir_inprogress, subject_names[i]),
                                                                verbose=False)
    
    # Resample the data to 256 Hz & convert to dataframe
    epochs = epochs.resample(sfreq=256)
    df_epochs = epochs.to_data_frame()

    ### Lempel-Ziv complexity

    # Go through all the channels signals
    lzc_i = []
    for ch in epochs.info['ch_names']:
        # Go through all epochs in the current channel signal
        lzc_ch = []
        for epo in df_epochs['epoch'].unique():
            # Calculate Lempel-Ziv Complexity (LZC) for the current epoch
            epo_signal = df_epochs[df_epochs['epoch']==epo][ch]

            scale = len(epo_signal) / (3 + 10)

            lzc_epo, info = nk.complexity_lempelziv(epo_signal, **lzc_args)
            lzc_ch.append(lzc_epo)
        # Average all epochs' LZC values to get a single value for the channel & add to list
        lzc_i.append(np.mean(lzc_ch))
    # Average all the channels' LZC values to get a single value for the subject & add to master dataframe
    lzc_i_mean = np.mean(lzc_i)
    df.loc[subject_names[i], 'LZC'] = lzc_i_mean

    ### Multiscale Sample Entropy

    # Go through all the channels signals
    mse_i = []
    mse_vals_i = np.zeros(shape=(len(epochs.info['ch_names']), mse_args['scale']))

    for c, ch in enumerate(epochs.info['ch_names']):
        # Go through all epochs in the current channel signal
        mse_ch = []
        mse_vals_epo = []
        for epo in df_epochs['epoch'].unique():
            # Calculate Multiscale Sample Entropy (MSE) measures for the current epoch
            epo_signal = df_epochs[df_epochs['epoch']==epo][ch]
            mse_epo, info = nk.entropy_multiscale(epo_signal.to_numpy(), **mse_args)
            # Get the total and scales' MSE values for the current epoch & add to list including all epochs
            mse_ch.append(mse_epo)
            mse_vals_epo.append(info.get('Value'))
        # Average all epochs' MSE values for every channel for the subject
        mse_vals_i[c] = np.mean(mse_vals_epo, axis=0)
        # Average all epochs' MSE totals to get a single value for the channel & add to list
        mse_i.append(np.mean(mse_ch))

    # Average all the channels' MSE totals & values to get global value
    mse_i_mean = np.mean(mse_i)
    mse_vals_i_mean = np.mean(mse_vals_i, axis=0)


    # Add total MSE to dataframe for the subject
    df.loc[subject_names[i], 'MSE (total)'] = mse_i_mean


    # Add all scales' MSE values to dataframe for the subject
    for scl in range(mse_args['scale']):
        df.loc[subject_names[i], 'MSE (scale={})'.format(scl+1)] = mse_vals_i_mean[scl]

display(df)
