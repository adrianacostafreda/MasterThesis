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

epoch_duration = 1

def characterization_eeg(raw, epoch_duration):
    
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
            
            duration_trigger = annot.duration[idx] + 60 # duration of 1, 2, 3 back is 60 s
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

    raw_0back = raw_new.copy().crop(tmin=trigger_data['4']['begin'][0], tmax=trigger_data['4']['end'][0] + 11)
    raw_1back = raw_new.copy().crop(tmin=trigger_data['5']['begin'][0], tmax=trigger_data['5']['end'][0] + 11)
    raw_2back = raw_new.copy().crop(tmin=trigger_data['6']['begin'][0], tmax=trigger_data['6']['end'][0] + 11)
    raw_3back = raw_new.copy().crop(tmin=trigger_data['7']['begin'][0], tmax=trigger_data['7']['end'][0] + 1)

    #print("eeg 0back", trigger_data['4']['end'][0] + 10 - trigger_data['4']['begin'][0])
    #print("eeg 1back", trigger_data['5']['end'][0] + 10 - trigger_data['5']['begin'][0])
    #print("eeg 2back", trigger_data['6']['end'][0] + 10 - trigger_data['6']['begin'][0])
    #print("eeg 3back", trigger_data['7']['end'][0] - trigger_data['7']['begin'][0])

    # Epoch data

    """
    Returns a mne.Epochs object created from the annotated mne.io.Raw object.

    """

    eeg_epochs_coupling = list()

    # make a 10 s delay
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

    eeg_epochs_coupling.append(epochs_0back_list)
    eeg_epochs_coupling.append(epochs_1back_list)
    eeg_epochs_coupling.append(epochs_2back_list)
    eeg_epochs_coupling.append(epochs_3back_list)

    #print("This is the length eeg_epochs", len(eeg_epochs_coupling))
    #print("This is the length eeg_epochs", eeg_epochs_coupling)

    return eeg_epochs_coupling

def bandpower_from_psd_ndarray(psd, freqs, band, ln_normalization = False, relative = True):
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

# Set default directory
os.chdir("/Users/adriana/Documents/GitHub/thesis")
#os.chdir("H:\Dokumenter\GitHub\MasterThesis\.venv")
mne.set_log_level('error')

# Folder where to get the raw EEG files
#clean_raw_eeg_folder = "H:\\Dokumenter\\data_acquisition\\data_eeg\\clean_eeg\\healthy_controls\\first_trial\\"
clean_raw_eeg_hc = "/Users/adriana/Documents/DTU/thesis/data_acquisition/clean_eeg/healthy_controls/"
clean_raw_eeg_p = "/Users/adriana/Documents/DTU/thesis/data_acquisition/clean_eeg/patients/"

# Get directories of raw EEG files and set export directory for clean files
eeg_dir_inprogress_hc = os.path.join(clean_raw_eeg_hc)
eeg_dir_inprogress_p = os.path.join(clean_raw_eeg_p)

file_dirs_eeg_hc, hc_eeg = read_files(eeg_dir_inprogress_hc,'.fif')
file_dirs_eeg_p, p_eeg = read_files(eeg_dir_inprogress_p,'.fif')

bands = [(0.5, 4, 'Delta'), (4, 8, 'Theta'), (8, 12, 'Alpha'), 
        (12, 16, 'Sigma'), (16, 30, 'Beta')]

mne.set_config('MNE_BROWSER_BACKEND', 'matplotlib')

delay_time = np.arange(0, 10, epoch_duration)

# Healthy controls
subjects_hc = list()

# Loop through all the subjects' directories (EEG files directories)
for file_hc in range(len(file_dirs_eeg_hc)):
    
    # --------------Read Raw data -------------------
    raw_hc = mne.io.read_raw_fif(file_dirs_eeg_hc[file_hc])
    raw_hc.load_data()

    # --------------Epoched Data--------------------------------
    # Each epoched object contains 1s epoched data corresponding to each data segment

    epochs_eeg_hc = characterization_eeg(raw_hc, epoch_duration)
    #print("This is the list containing the epochs", len(epochs_eeg_hc))

    hc_bp_relative_nback = list()

    # --------------Extract features--------------------------------
    for nback_hc in epochs_eeg_hc:

        hc_coupling_list = list()

        for coupling_hc in range(len(nback_hc)):
            #print("This is each segment of the epoched data with delay", coupling[i])
            
            channel_drop = [ 'AF7', 'AFF5h', 'FT7', 'FC5', 'FC3', 'FCC3h', 
                    'CCP3h', 'CCP1h', 'CP1', 'TP7', 'CPP3h', 'P1', 
                    'AF8', 'AFF6h', 'FT8', 'FC6', 'FC4', 'FCC4h', 'CCP4h', 
                    'CCP2h', 'CP2', 'P2', 'CPP4h', 'TP8']

            nback_hc[coupling_hc].drop_channels(channel_drop)

            # Create a 3-D array
            data = nback_hc[coupling_hc].get_data(units="uV")
            sf = nback_hc[coupling_hc].info['sfreq']

            #print("This is the shape of data", data.shape)

            win = int(4 * sf)  # Window size is set to 4 seconds

            # Compute the relative power
            freqs, psd = welch(data, sf, nperseg=win) 
            bp_relative = bandpower_from_psd_ndarray(psd, freqs, bands, ln_normalization = False, relative = True)
            #print("This is the shape of bp_relative for hc", bp_relative.shape)
           
            hc_coupling_list.append(bp_relative)
            #print("This is the length of nback list", len(coupling_list))

        hc_bp_relative_nback.append(hc_coupling_list)
    
    #print("This is the length of coupling list", len(hc_bp_relative_nback))

    nback_hc = list()

    for n_back_hc in hc_bp_relative_nback:
        #print("this is the length n_back", len(n_back))
        
        hc_coupling_list_bp = list()

        for coupling_hc in n_back_hc:
            
            coupling_hc = np.expand_dims(coupling_hc, axis=0)
            #print("This is the shape of coupling", coupling.shape)
            hc_coupling_list_bp.append(coupling_hc)

        nback_hc.append(hc_coupling_list_bp)

    subjects_hc.append(nback_hc)

print("This is the shape hc", len(subjects_hc[0]))

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
    #print(f"Delay {delay_index_hc} concatenated arrays:")
    for shape_hc, array_hc in subject_hc.items():
        
        if delay_index_hc == 0:
            #print(f"Shape 0back {shape_hc}: {array_hc.shape}")
            delays_0back_hc.append(array_hc)
        
        if delay_index_hc == 1:
            #print(f"Shape 1back {shape_hc}: {array_hc.shape}")
            delays_1back_hc.append(array_hc)
        
        if delay_index_hc == 2:
            #print(f"Shape 2back {shape_hc}: {array_hc.shape}")
            delays_2back_hc.append(array_hc)
        
        if delay_index_hc == 3:
            #print(f"Shape 3back {shape_hc}: {array_hc.shape}")
            delays_3back_hc.append(array_hc)
                
#print("This is the length of 0back", len(delays_0back_hc))
#print("This is the length of 1back", len(delays_1back_hc))
#print("This is the length of 2back", len(delays_2back_hc))
#print("This is the length of 3back", len(delays_3back_hc))

# 0 back
hc_mean_0back = list()
for delay in delays_0back_hc:
    #print("This is the shape", delay.shape)
    theta_bp_relative_hc = delay[:, 1, : , :] #(subj, freq band, epochs, channels)
    mean_hc = np.mean(theta_bp_relative_hc, axis=0)
    #print("This is the shape of mean_hc", mean_hc.shape)
    mean_channels_hc = np.mean(mean_hc, axis=-1)
    #print("This is the shape of mean_channels_hc", mean_channels_hc.shape)
    mean_epochs_hc = np.mean(mean_channels_hc, axis=0)
    #print("This is mean_epochs_hc", mean_epochs_hc)
    hc_mean_0back.append(mean_epochs_hc)

print("This is the length of hc_mean 0back ", hc_mean_0back)

# 1 back
hc_mean_1back = list()
for delay in delays_1back_hc:
    #print("This is the shape", delay.shape)
    theta_bp_relative_hc = delay[:, 1, : , :] #(subj, freq band, epochs, channels)
    mean_hc = np.mean(theta_bp_relative_hc, axis=0)
    #print("This is the shape of mean_hc", mean_hc.shape)
    mean_channels_hc = np.mean(mean_hc, axis=-1)
    #print("This is the shape of mean_channels_hc", mean_channels_hc.shape)
    mean_epochs_hc = np.mean(mean_channels_hc, axis=0)
    #print("This is mean_epochs_hc", mean_epochs_hc)
    hc_mean_1back.append(mean_epochs_hc)

print("This is the length of hc_mean 1back", hc_mean_1back)

# 2 back
hc_mean_2back = list()
for delay in delays_2back_hc:
    #print("This is the shape", delay.shape)
    theta_bp_relative_hc = delay[:, 1, : , :] #(subj, freq band, epochs, channels)
    mean_hc = np.mean(theta_bp_relative_hc, axis=0)
    #print("This is the shape of mean_hc", mean_hc.shape)
    mean_channels_hc = np.mean(mean_hc, axis=-1)
    #print("This is the shape of mean_channels_hc", mean_channels_hc.shape)
    mean_epochs_hc = np.mean(mean_channels_hc, axis=0)
    #print("This is mean_epochs_hc", mean_epochs_hc)
    hc_mean_2back.append(mean_epochs_hc)

print("This is the length of hc_mean 2back", hc_mean_2back)

# 3 back
hc_mean_3back = list()
for delay in delays_3back_hc:
    #print("This is the shape", delay.shape)
    theta_bp_relative_hc = delay[:, 1, : , :] #(subj, freq band, epochs, channels)
    mean_hc = np.mean(theta_bp_relative_hc, axis=0)
    #print("This is the shape of mean_hc", mean_hc.shape)
    mean_channels_hc = np.mean(mean_hc, axis=-1)
    #print("This is the shape of mean_channels_hc", mean_channels_hc.shape)
    mean_epochs_hc = np.mean(mean_channels_hc, axis=0)
    #print("This is mean_epochs_hc", mean_epochs_hc)
    hc_mean_3back.append(mean_epochs_hc)

print("This is the length of hc_mean 3back", hc_mean_3back)

# Patients
subjects_p = list()

# Loop through all the subjects' directories (EEG files directories)
for file_p in range(len(file_dirs_eeg_p)):
    
    # --------------Read Raw data -------------------
    raw_p = mne.io.read_raw_fif(file_dirs_eeg_p[file_p])
    raw_p.load_data()

    # --------------Epoched Data--------------------------------
    # Each epoched object contains 1s epoched data corresponding to each data segment

    epochs_eeg_p = characterization_eeg(raw_p, epoch_duration)
    #print("This is the list containing the epochs", len(epochs_eeg_hc))

    p_bp_relative_nback = list()

    # --------------Extract features--------------------------------
    for nback_p in epochs_eeg_p:

        p_coupling_list = list()

        for coupling_p in range(len(nback_p)):
            #print("This is each segment of the epoched data with delay", coupling[i])
            
            channel_drop = [ 'AF7', 'AFF5h', 'FT7', 'FC5', 'FC3', 'FCC3h', 
                    'CCP3h', 'CCP1h', 'CP1', 'TP7', 'CPP3h', 'P1', 
                    'AF8', 'AFF6h', 'FT8', 'FC6', 'FC4', 'FCC4h', 'CCP4h', 
                    'CCP2h', 'CP2', 'P2', 'CPP4h', 'TP8']

            nback_p[coupling_p].drop_channels(channel_drop)

            # Create a 3-D array
            data = nback_p[coupling_p].get_data(units="uV")
            sf = nback_p[coupling_p].info['sfreq']

            #print("This is the shape of data", data.shape)

            win = int(4 * sf)  # Window size is set to 4 seconds

            # Compute the relative power
            freqs, psd = welch(data, sf, nperseg=win) 
            bp_relative = bandpower_from_psd_ndarray(psd, freqs, bands, ln_normalization = False, relative = True)
            #print("This is the shape of bp_relative for hc", bp_relative.shape)
           
            p_coupling_list.append(bp_relative)
            #print("This is the length of nback list", len(p_coupling_list))

        p_bp_relative_nback.append(p_coupling_list)
    
    #print("This is the length of coupling list", len(p_bp_relative_nback))

    nback_p = list()

    for n_back_p in p_bp_relative_nback:
        #print("this is the length n_back", len(n_back_p))
        
        p_coupling_list_bp = list()

        for coupling_p in n_back_p:
            
            coupling_p = np.expand_dims(coupling_p, axis=0)
            #print("This is the shape of coupling", coupling_p.shape)
            p_coupling_list_bp.append(coupling_p)

        nback_p.append(p_coupling_list_bp)

    subjects_p.append(nback_p)

print("This is the shape of subjects_p", len(subjects_p[0]))

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
for delay_p in delays_0back_p:
    theta_bp_relative_p = delay_p[:, 1, : , :] #(subj, freq band, epochs, channels)
    mean_p = np.mean(theta_bp_relative_p, axis=0)
    mean_channels_p = np.mean(mean_p, axis=-1)
    mean_epochs_p = np.mean(mean_channels_p, axis=0)
    p_mean_0back.append(mean_epochs_p)

print("This is the length of p_mean 0back ", p_mean_0back)

# 1 back
p_mean_1back = list()
for delay_p in delays_1back_p:   
    theta_bp_relative_p = delay_p[:, 1, : , :] #(subj, freq band, epochs, channels)
    mean_p = np.mean(theta_bp_relative_p, axis=0)  
    mean_channels_p = np.mean(mean_p, axis=-1)    
    mean_epochs_p = np.mean(mean_channels_p, axis=0)
    p_mean_1back.append(mean_epochs_p)

print("This is the length of p_mean 1back", p_mean_1back)

# 2 back
p_mean_2back = list()
for delay_p in delays_2back_p:
    theta_bp_relative_p = delay_p[:, 1, : , :] #(subj, freq band, epochs, channels)
    mean_p = np.mean(theta_bp_relative_p, axis=0)
    mean_channels_p = np.mean(mean_p, axis=-1)
    mean_epochs_p = np.mean(mean_channels_p, axis=0)
    p_mean_2back.append(mean_epochs_p)

print("This is the length of p_mean 2back", p_mean_2back)

# 3 back
p_mean_3back = list()
for delay_p in delays_3back_p:  
    theta_bp_relative_p = delay[:, 1, : , :] #(subj, freq band, epochs, channels)
    mean_p = np.mean(theta_bp_relative_p, axis=0)   
    mean_channels_p = np.mean(mean_p, axis=-1) 
    mean_epochs_p = np.mean(mean_channels_p, axis=0)
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

"""
#for idx, lista in enumerate(preprocess_eeg):
    #print(f"This is the length of each individual list in preprocess eeg {idx+1}: {len(lista)}")

#eeg_bandpower = "H:\\Dokumenter\\data_processing\\neurovascular_coupling\\EEG_bandpower\\"
eeg_bandpower = "/Users/adriana/Documents/DTU/thesis/data_processing/neurovascular_coupling/EEG_bandpower/"

empty_directory(eeg_bandpower)

for subject_name in subject_names_eeg:
    path_eeg_bandpower = '{}/{}/'.format(eeg_bandpower, subject_name)
    os.makedirs(path_eeg_bandpower, exist_ok=True)
    
    for subject in preprocess_eeg:
        delay_count = 0
        for bp_delay in subject:
            #print("This is the shape of coupling (frequency band, epochs, channels)", bp_delay.shape)
            np.save(path_eeg_bandpower + "coupling" + str(delay_count), bp_delay)
            delay_count = delay_count + 1 

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
raw_fnirs_folder = "/Users/adriana/Documents/DTU/thesis/data_acquisition/data_fnirs/healthy_controls/"

# Get directories of raw EEG files and set export directory for clean files
fnirs_dir_inprogress = os.path.join(raw_fnirs_folder)

file_dirs_fnirs, subject_names_fnirs = read_files(fnirs_dir_inprogress,'.snirf')

preprocess_fnirs = list()

# Loop over all files preprocess them with HemoData and save them at an numpy array.
# It save the data for each patient, for each event, for each channel:  patient X events X channels X time_samples.

for i in range(len(file_dirs_fnirs)):

    # --------------Read Raw data -------------------
    # --------------Preprocess of fNIRS Raw data -------------------

    raw_haemo = HemoData(file_dirs_fnirs[i], preprocessing=True, isPloting=False).getMneIoRaw()
    #raw_haemo.plot(block=True)

    epochs_fnirs = characterization_fNIRS(raw_haemo, epoch_duration)

    coupling_data_fnirs = list()
    # --------------Extract features-------------------------------- 
    for coupling in epochs_fnirs:

        concatenated_data = list()

        for i in range(len(coupling)):
            #print("This is each segment of the epoched data with delay", coupling[i])
            
            raw_data_hbo = coupling[i].get_data(picks=["hbo"]) # epochs x channels x samples
            
            #print("This is the shape for raw_data_hbo", raw_data_hbo.shape)
            
            # Append the data array to the list
            concatenated_data.append(raw_data_hbo)

        concatenated_hbo = np.concatenate(concatenated_data, axis = 0)

        coupling_data_fnirs.append(concatenated_hbo)

    #print("This is the list of coupling_data", coupling_data_fnirs)
    #print("This is the length of coupling data", len(coupling_data_fnirs))

    preprocess_fnirs.append(coupling_data_fnirs)



#fnirs_samples = "H:\\Dokumenter\\data_processing\\neurovascular_coupling\\fnirs_samples\\"
fnirs_samples = "/Users/adriana/Documents/DTU/thesis/data_processing/neurovascular_coupling/fnirs_samples/"

empty_directory(fnirs_samples)

for subject_name in subject_names_fnirs:
    path_fnirs_samples = '{}/{}/'.format(fnirs_samples, subject_name)
    os.makedirs(path_fnirs_samples, exist_ok=True)
    
    for subject in preprocess_fnirs:
        delay_count = 0
        for fnirs_coupling in subject:
            #print("This is the shape of fnirs_coupling (epochs, channels, samples) ", fnirs_coupling.shape)
            np.save(path_fnirs_samples + "coupling" + str(delay_count), fnirs_coupling)
            delay_count = delay_count + 1 

#fnirs_features = "H:\\Dokumenter\\data_processing\\neurovascular_coupling\\fnirs_features\\"
fnirs_features = "/Users/adriana/Documents/DTU/thesis/data_processing/neurovascular_coupling/fnirs_features/"

empty_directory(fnirs_features)

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