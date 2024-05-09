import os
import mne
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import simps
from scipy.signal import welch

import yasa

from IPython.display import display

# Set default directory
os.chdir("/Users/adriana/Documents/GitHub/thesis/MasterThesis/")
mne.set_log_level('error')

# Import functions
#import signal_processing.spectrum as spectrum
#import basic.arrange_files as arrange

def read_files(dir_inprogress,filetype,exclude_subjects=[],verbose=True):
    """
    Get all the (EEG) file directories and subject names.

    Parameters
    ----------
    dir_inprogress: A string with directory to look for files
    filetype: A string with the ending of the files we are looking for (e.g. '.xdf')

    Returns
    -------
    file_dirs: A list of strings with file directories for all the (EEG) files
    subject_names: A list of strings with all the corresponding subject names
    """

    file_dirs = []
    subject_names = []

    for file in os.listdir(dir_inprogress):
        if file.endswith(filetype):
            file_dirs.append(os.path.join(dir_inprogress, file))
            #subject_names.append(os.path.join(file).removesuffix(filetype))
            subject_names.append(file[:-len(filetype)])

    try:
        for excl_sub in exclude_subjects:
            for i in range(len(subject_names)):
                if excl_sub in subject_names[i]:
                    if verbose == True:
                        print('EXCLUDED SUBJECT: ',excl_sub,'in',subject_names[i],'at',file_dirs[i])
                    del subject_names[i]
                    del file_dirs[i]
                    break
    except:
        pass
    
    file_dirs = sorted(file_dirs)
    subject_names = sorted(subject_names)

    if verbose == True:
        print("Files in {} read in: {}".format(dir_inprogress,len(file_dirs)))

    return [file_dirs, subject_names]

def create_results_folders(exp_folder, exp_condition, exp_condition_nback, results_folder='Results', abs_psd=False,
                           rel_psd=False, fooof = False):
    """
    Dummy way to try to pre-create folders for PSD results before exporting them

    Parameters
    ----------
    exp_folder: A string with a relative directory to experiment folder (e.g. 'Eyes Closed\Baseline')
    """

    if abs_psd == True:
        try:
            os.makedirs(os.path.join('{}/{}/{}/Absolute PSD/{}'.format(results_folder, exp_folder, exp_condition, exp_condition_nback)))
        except FileExistsError:
            pass
        
    
    if rel_psd == True:
        try:
            os.makedirs(os.path.join('{}/{}/{}/Relative PSD/{}'.format(results_folder, exp_folder, exp_condition, exp_condition_nback)))
        except FileExistsError:
            pass
    
    if fooof== True:
        try:
            os.makedirs(os.path.join('{}/{}/{}/FOOOF/{}'.format(results_folder, exp_folder, exp_condition, exp_condition_nback)))
        except FileExistsError:
            pass
    
    try:
        os.makedirs(os.path.join('{}/{}'.format(results_folder, exp_folder)))
    except FileExistsError:
        pass

def bandpower_from_psd_ndarray(psd, freqs, band, subjectname,
                          ln_normalization = False, relative = True):
    """
    Find frequency band in interest for all the channels. 

    Parameters
    ----------
    psds: An array for power spectrum density values 
    freqs: An array for corresponding frequencies
    band: A list of lower and higher frequency for the frequency band in interest
    subjectname: A string for subject's name
    relative : boolean
        If True, bandpower is divided by the total power between the min and
        max frequencies defined in ``band`` (default 0.5 to 40 Hz).

    Returns
    --------
    psds_band_mean_ch : An array for a frequency band power values for all the channels 
    
    """

    # Type checks
    assert isinstance(band, list), "bands must be a list of tuple(s)"
    assert isinstance(relative, bool), "relative must be a boolean"

    # Safety checks
    freqs = np.asarray(freqs)
    psd = np.asarray(psd)
    assert freqs.ndim == 1, "freqs must be a 1-D array of shape (n_freqs,)"
    assert psd.shape[-1] == freqs.shape[-1], "n_freqs must be last axis of psd"

    # Extract frequencies of interest
    all_freqs = np.hstack([[b[0], b[1]] for b in band])
    fmin, fmax = min(all_freqs), max(all_freqs)
    idx_good_freq = np.logical_and(freqs >= fmin, freqs <= fmax)
    freqs = freqs[idx_good_freq]
    res = freqs[1] - freqs[0]

    # Trim PSD to frequencies of interest
    psd = psd[..., idx_good_freq]

    # Check if there are negative values in PSD
    if (psd < 0).any():
        msg = (
            "There are negative values in PSD. This will result in incorrect "
            "bandpower values. We highly recommend working with an "
            "all-positive PSD. For more details, please refer to: "
            "https://github.com/raphaelvallat/yasa/issues/29"
        )

    # Calculate total power
    total_power = simps(psd, dx=res, axis=-1)
    total_power = total_power[np.newaxis, ...]

    # Initialize empty array
    bp = np.zeros((len(band), *psd.shape[:-1]), dtype=np.float64)

    # Enumerate over the frequency bands
    labels = []
    for i, band in enumerate(bands):
        b0, b1, la = band
        labels.append(la)
        idx_band = np.logical_and(freqs >= b0, freqs <= b1)
        bp[i] = simps(psd[..., idx_band], dx=res, axis=-1)

    if relative:
        bp /= total_power
    
    # If true, normalize the BP with natural logarithm transform
    if ln_normalization == True:
        bp = np.log(bp)
    
    return bp

# Folder where to get the clean epochs files
clean_folder = "/Users/adriana/Documents/DTU/thesis/data_acquisition/clean_eeg_epoch/"

exp_folder = 'healthy_controls'
exp_condition = 'n_back'
exp_condition_nback = '2_back'
exp_condition_nback_num = 2

bands=[(0.5, 4, 'Delta'), (4, 8, 'Theta'), (8, 12, 'Alpha'), 
       (12, 16, 'Sigma'), (16, 30, 'Beta')]

# Get directories of clean EEG files and set export directory
dir_inprogress = os.path.join(clean_folder, exp_folder, exp_condition, exp_condition_nback)
file_dirs, subject_names = read_files(dir_inprogress,'_clean-epo.fif')

# Initialize an empty list to store bandpower arrays for all subjects
all_subject_bandpower_relative = []
all_subject_bandpower_absolute = []

for i in range(len(file_dirs)):
    # Read the clean data from the disk

    epochs = mne.read_epochs(fname='{}/{}_clean-epo.fif'.format(dir_inprogress, subject_names[i]),
                                                                verbose=False)
    
    channel_drop = [ 'AF7', 'AFF5h', 'FT7', 'FC5', 'FC3', 'FCC3h', 
                    'CCP3h', 'CCP1h', 'CP1', 'TP7', 'CPP3h', 'P1', 
                    'AF8', 'AFF6h', 'FT8', 'FC6', 'FC4', 'FCC4h', 'CCP4h', 
                    'CCP2h', 'CP2', 'P2', 'CPP4h', 'TP8']

    epochs = epochs.drop_channels(channel_drop)

    print("These are the channel names", epochs.ch_names)

    # Create a 3-D array
    data = epochs.get_data(units="uV")
    sf = epochs.info['sfreq']

    win = int(4 * sf)  # Window size is set to 4 seconds
    freqs, psd = welch(data, sf, nperseg=win, axis=-1) 

    # Relative power 
    bandpower_relative = bandpower_from_psd_ndarray(psd, freqs, bands, subject_names[i],   
                        ln_normalization = False, relative = True)
    
    # Add an additional dimension to the bandpower array
    bandpower_relative = np.expand_dims(bandpower_relative, axis=0)

    # Append the bandpower array to the list
    all_subject_bandpower_relative.append(bandpower_relative)

    # Absolute power 
    bandpower_absolute = bandpower_from_psd_ndarray(psd, freqs, bands, subject_names[i],   
                        ln_normalization = False, relative = False)
    
    # Add an additional dimension to the bandpower array
    bandpower_absolute = np.expand_dims(bandpower_absolute, axis=0)

    # Append the bandpower array to the list
    all_subject_bandpower_absolute.append(bandpower_absolute)

# Concatenate all bandpower arrays along the first axis to create a 4-D array
all_subject_bandpower_relative = np.concatenate(all_subject_bandpower_relative, axis=0)
print(all_subject_bandpower_relative.shape)

# Concatenate all bandpower arrays along the first axis to create a 4-D array
all_subject_bandpower_absolute = np.concatenate(all_subject_bandpower_absolute, axis=0)
print(all_subject_bandpower_absolute.shape)

# Save data path
results_foldername = "/Users/adriana/Documents/DTU/thesis/data_processing/Results_EEG"

# Pre-create results folders for spectral analysis data
create_results_folders(results_folder=results_foldername, exp_folder=exp_folder, exp_condition=exp_condition,
                       exp_condition_nback=exp_condition_nback, abs_psd=False, rel_psd=True)

path_absolute = '{}/{}/{}/Absolute PSD/{}/'.format(results_foldername, exp_folder, exp_condition, exp_condition_nback)

# Create the directory if it doesn't exist
os.makedirs(path_absolute, exist_ok=True)

# Save the array to a file
np.save(os.path.join(path_absolute, str(exp_condition_nback_num)), all_subject_bandpower_absolute)

# Pre-create results folders for spectral analysis data
create_results_folders(results_folder=results_foldername, exp_folder=exp_folder, exp_condition=exp_condition,
                       exp_condition_nback=exp_condition_nback, abs_psd=False, rel_psd=True,
                       fooof = False)

path_relative = '{}/{}/{}/Relative PSD/{}/'.format(results_foldername, exp_folder, exp_condition, exp_condition_nback)

# Create the directory if it doesn't exist
os.makedirs(path_relative, exist_ok=True)

# Save the array to a file
np.save(os.path.join(path_relative, str(exp_condition_nback_num)), all_subject_bandpower_relative)