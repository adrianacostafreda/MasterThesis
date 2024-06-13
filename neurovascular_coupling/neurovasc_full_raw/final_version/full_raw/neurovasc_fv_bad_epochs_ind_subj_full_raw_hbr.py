import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
from scipy.integrate import simps
from scipy.signal import welch
from autoreject import get_rejection_threshold
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from basic.arrange_files import read_files
from neurovascular_coupling.Hemo import HemoData
#from arrange_files import read_files
#from Hemo import HemoData

"""

Characterize data

"""

# Function to visualize dropped epochs
def visualize_dropped_epochs(epochs):

    dropped_epochs_indices = []

    dropped_epochs_ind = epochs.drop_log
    dropped_indices = [i for i, log in enumerate(dropped_epochs_ind) if len(log) > 0]
    dropped_epochs_indices.append(dropped_indices)
    
    return dropped_epochs_indices


# Function to characterize EEG data
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

    begin_trigger4 = trigger_data["5"]["begin"][0] - 15 - 48 - 4
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

    raw = raw_new.copy().crop(tmin=trigger_data['4']['begin'][0], tmax=trigger_data['7']['end'][0])

    # Epoch data

    """
    Returns a mne.Epochs object created from the annotated mne.io.Raw object.

    """

    eeg_bad_epochs = list()

    events = mne.make_fixed_length_events(raw, start = 0 , stop = raw.times[-1], duration = epoch_duration)
    
    interval = (0,0)
    epochs_eeg = mne.Epochs(raw, events, baseline = interval, preload = True)
        
    # -----------------Get-Rejection-Threshold---------------------------------
    reject_thresholds = get_rejection_threshold(epochs_eeg, ch_types = "eeg", verbose = False)
        
    epochs_eeg.drop_bad(reject=reject_thresholds)
    
    # Visualize dropped epochs
    bad_epoch = visualize_dropped_epochs(epochs_eeg)
        
    eeg_bad_epochs.append(bad_epoch)

    return (epochs_eeg, eeg_bad_epochs)

def characterization_fNIRS(raw, epoch_duration, bad_epochs):
    
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

    begin_trigger4 = trigger_data["5"]["begin"][0] - 15 - 48 - 4
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

    raw = raw_new.copy().crop(tmin=trigger_data['4']['begin'][0], tmax=trigger_data['7']['end'][0])

    # Epoch data

    for bad in bad_epochs:
        events= mne.make_fixed_length_events(raw, start = 0, stop= raw.times[-1], duration = epoch_duration)
        
        interval = (0,0)
        fnirs_epochs = mne.Epochs(raw, events, baseline = interval, preload = True)
        
        # Convert one-based indices to zero-based indices
        bad_zero_based = [i - 1 for i in bad[0]]
        
        # Check if each bad index is within the range of available epochs
        valid_bad = [idx for idx in bad_zero_based if idx in fnirs_epochs.selection - 1]

        # Drop the valid bad epochs
        if valid_bad:
            fnirs_epochs.drop(indices=valid_bad)

    return fnirs_epochs

"""

Compute relative bandpower

"""

# Function to calculate bandpower from PSD ndarray
def bandpower_from_psd_ndarray(psd, freqs, bands, ln_normalization=False, relative=True):
    
    freqs = np.asarray(freqs)
    psd = np.asarray(psd)
    all_freqs = np.hstack([[b[0], b[1]] for b in bands])
    fmin, fmax = min(all_freqs), max(all_freqs)
    idx_good_freq = np.logical_and(freqs >= fmin, freqs <= fmax)
    freqs = freqs[idx_good_freq]
    res = freqs[1] - freqs[0]
    psd = psd[..., idx_good_freq]

    total_power = simps(psd, dx=res, axis=-1)[np.newaxis, ...]

    bp = np.zeros((len(bands), *psd.shape[:-1]), dtype=np.float64)
    for i, band in enumerate(bands):
        b0, b1, _ = band
        idx_band = np.logical_and(freqs >= b0, freqs <= b1)
        bp[i] = simps(psd[..., idx_band], dx=res, axis=-1)

    if relative:
        bp /= total_power

    if ln_normalization:
        bp = np.log(bp)

    return bp

"""

Process data

"""

# fNIRS

def process_fnirs_data(file_dirs_fnirs, fnirs, epoch_duration, bad_epochs):
    
    subjects = []
    for file_dir, bad_epoch in zip(file_dirs_fnirs, bad_epochs):
        print("File dir of fNIRS", file_dir)
        
        raw_haemo = HemoData(file_dir, preprocessing=True, isPloting=False).getMneIoRaw()

        epochs_fnirs = characterization_fNIRS(raw_haemo, epoch_duration, bad_epoch)

        features_fnirs = []
        raw_hbo = epochs_fnirs.get_data(picks=[fnirs])
        mean_std_per_epoch_per_channel = np.concatenate([
                np.expand_dims(np.mean(raw_hbo, axis=-1), axis=-1),
                np.expand_dims(np.std(raw_hbo, axis=-1), axis=-1)], axis=-1)

        features_fnirs.append(mean_std_per_epoch_per_channel)
        
        fnirs_data = np.concatenate(features_fnirs, axis=0)
        subjects.append(fnirs_data)

    return subjects

# Set up directories
#os.chdir("/Users/adriana/Documents/GitHub/thesis")
os.chdir("H:\Dokumenter\GitHub\MasterThesis\.venv")
mne.set_log_level('error')
mne.set_config('MNE_BROWSER_BACKEND', 'matplotlib')

# EEG
#clean_raw_eeg_hc = "/Users/adriana/Documents/DTU/thesis/data_acquisition/clean_eeg/healthy_controls/"
#clean_raw_eeg_p = "/Users/adriana/Documents/DTU/thesis/data_acquisition/clean_eeg/patients/"

clean_raw_eeg_hc = "H:\\Dokumenter\\data_acquisition\\data_eeg\\clean_eeg\\healthy_controls\\"
clean_raw_eeg_p = "H:\\Dokumenter\\data_acquisition\\data_eeg\\clean_eeg\\patients\\"

# fNIRS
#clean_raw_fnirs_hc = "/Users/adriana/Documents/DTU/thesis/data_acquisition/data_fnirs/healthy_controls/"
#clean_raw_fnirs_p = "/Users/adriana/Documents/DTU/thesis/data_acquisition/data_fnirs/patients/"

clean_raw_fnirs_hc = "H:\\Dokumenter\\data_acquisition\\data_fnirs\\healthy_controls\\baseline\\snirf_files\\"
clean_raw_fnirs_p = "H:\\Dokumenter\\data_acquisition\\data_fnirs\\patients\\baseline\\snirf_files\\"

# EEG
file_dirs_eeg_hc, hc_eeg = read_files(clean_raw_eeg_hc, '.fif')
file_dirs_eeg_p, p_eeg = read_files(clean_raw_eeg_p, '.fif')

# fNIRS
file_dirs_fnirs_hc, _ = read_files(clean_raw_fnirs_hc, '.snirf')
file_dirs_fnirs_p, _ = read_files(clean_raw_fnirs_p, '.snirf')

bands = [(0.5, 4, 'Delta'), (4, 8, 'Theta'), (8, 12, 'Alpha'), 
         (12, 16, 'Sigma'), (16, 30, 'Beta')]

epoch_duration = 1

"""
EEG
"""

# HEALTHY CONTROLS
subjects_eeg_hc = []
subj_hc_bad_epochs = []

for file_hc in file_dirs_eeg_hc:
    print("This is the file_hc", file_hc)
    raw = mne.io.read_raw_fif(file_hc)
    raw.load_data()
    
    characterize_eeg_hc = characterization_eeg(raw, epoch_duration)

    epochs_hc = characterize_eeg_hc[0]
    bad_epochs_hc = characterize_eeg_hc[1]

    channel_drop = [ 'AF7', 'AFF5h', 'FT7', 'FC5', 'FC3', 'FCC3h', 
                    'CCP3h', 'CCP1h', 'CP1', 'TP7', 'CPP3h', 'P1', 
                    'AF8', 'AFF6h', 'FT8', 'FC6', 'FC4', 'FCC4h', 'CCP4h', 
                    'CCP2h', 'CP2', 'P2', 'CPP4h', 'TP8']
        
    epochs_hc.drop_channels(channel_drop)
    data_hc = epochs_hc.get_data(units="uV")

    sf = epochs_hc.info['sfreq']
        
    nperseg = 256
    noverlap = 0.5*256
    freqs_hc, psd_hc = welch(data_hc, sf, nperseg=256, noverlap=0.5*256)
    bp = bandpower_from_psd_ndarray(psd_hc, freqs_hc, bands, ln_normalization=True, relative=True)

    subjects_eeg_hc.append(bp)
    subj_hc_bad_epochs.append(bad_epochs_hc)


# PATIENTS

subjects_eeg_p = []
subj_p_bad_epochs = []

for file_p in file_dirs_eeg_p:
    print("This is the file_p", file_p)
    raw = mne.io.read_raw_fif(file_p)
    raw.load_data()
    
    characterize_eeg_p = characterization_eeg(raw, epoch_duration)

    epochs_p = characterize_eeg_p[0]
    bad_epochs_p = characterize_eeg_p[1]
        
    channel_drop = [ 'AF7', 'AFF5h', 'FT7', 'FC5', 'FC3', 'FCC3h', 
                    'CCP3h', 'CCP1h', 'CP1', 'TP7', 'CPP3h', 'P1', 
                    'AF8', 'AFF6h', 'FT8', 'FC6', 'FC4', 'FCC4h', 'CCP4h', 
                    'CCP2h', 'CP2', 'P2', 'CPP4h', 'TP8']
    epochs_p.drop_channels(channel_drop)

    data_p = epochs_p.get_data(units="uV")
    sf = epochs_p.info['sfreq']

    nperseg = 256
    noverlap = 0.5*256
    freqs_p, psd_p = welch(data_p, sf, nperseg=256, noverlap=0.5*256)
    bp = bandpower_from_psd_ndarray(psd_p, freqs_p, bands, ln_normalization=True, relative=True)

    subjects_eeg_p.append(bp)
    subj_p_bad_epochs.append(bad_epochs_p)

fnirs_delays_hbo_hc = process_fnirs_data(file_dirs_fnirs_hc, "hbo", epoch_duration, subj_hc_bad_epochs)
fnirs_delays_hbo_p = process_fnirs_data(file_dirs_fnirs_p, "hbo", epoch_duration, subj_p_bad_epochs)

fnirs_delays_hbr_hc = process_fnirs_data(file_dirs_fnirs_hc, "hbr", epoch_duration, subj_hc_bad_epochs)
fnirs_delays_hbr_p = process_fnirs_data(file_dirs_fnirs_p, "hbr", epoch_duration, subj_p_bad_epochs)

# Number of shifts and increment
num_shifts = 30
increment = 0.5

# Calculate the maximum shift amount in terms of array indices
max_shift = int(num_shifts * increment)

# Create lists to store the shifted EEG arrays and corresponding reduced fNIRS arrays

hc_subj_shifted_EEG_arrays = []
hc_subj_reduced_hbo_fnirs_arrays = []
hc_subj_reduced_hbr_fnirs_arrays = []

for subject_eeg_hc, subject_fnirs_hbo_hc, subject_fnirs_hbr_hc in zip(subjects_eeg_hc, fnirs_delays_hbo_hc, fnirs_delays_hbr_hc):
    shifted_EEG_arrays_hc = []
    reduced_hbo_fnirs_arrays_hc = []
    reduced_hbr_fnirs_arrays_hc = []

    for shift in range(1, num_shifts + 1):
        shift_amount = int(shift * increment)
    
        # Shift EEG array to the right without filling with zeros by slicing
        if shift_amount < subject_eeg_hc.shape[1]:
            shifted_EEG = subject_eeg_hc[:, shift_amount:, :]
              
        else:
            shifted_EEG = np.array([])  # If shift_amount exceeds the array length, use an empty array

        # Reduce the corresponding portion from the fNIRS array
        reduced_fnirs_hbo = subject_fnirs_hbo_hc[:-shift_amount, :, :]
        reduced_fnirs_hbr = subject_fnirs_hbr_hc[:-shift_amount, :, :]

        # Ensure both arrays have compatible shapes
        if shifted_EEG.shape[1] == reduced_fnirs_hbo.shape[0]:
            shifted_EEG_arrays_hc.append(shifted_EEG)
            reduced_hbo_fnirs_arrays_hc.append(reduced_fnirs_hbo)
        else:
            # In case of dimension mismatch, adjust the lengths to match
            min_length = min(shifted_EEG.shape[1], reduced_fnirs_hbo.shape[0])
            if min_length > 0:
                shifted_EEG_arrays_hc.append(shifted_EEG[:, :min_length, :])
                reduced_hbo_fnirs_arrays_hc.append(reduced_fnirs_hbo[:min_length, :, :])
            else:
                # Handle the case where resulting arrays would be empty
                shifted_EEG_arrays_hc.append(np.array([]))
                reduced_hbo_fnirs_arrays_hc.append(np.array([]))
        
        # Ensure both arrays have compatible shapes
        if shifted_EEG.shape[1] == reduced_fnirs_hbr.shape[0]:
            shifted_EEG_arrays_hc.append(shifted_EEG)
            reduced_hbr_fnirs_arrays_hc.append(reduced_fnirs_hbr)
        else:
            # In case of dimension mismatch, adjust the lengths to match
            min_length = min(shifted_EEG.shape[1], reduced_fnirs_hbr.shape[0])
            if min_length > 0:
                shifted_EEG_arrays_hc.append(shifted_EEG[:, :min_length, :])
                reduced_hbr_fnirs_arrays_hc.append(reduced_fnirs_hbr[:min_length, :, :])
            else:
                # Handle the case where resulting arrays would be empty
                shifted_EEG_arrays_hc.append(np.array([]))
                reduced_hbr_fnirs_arrays_hc.append(np.array([]))
    
    # Example of how to access the shifted arrays and corresponding reduced arrays
    #for i, (shifted_EEG, reduced_fnirs) in enumerate(zip(shifted_EEG_arrays_hc, reduced_fnirs_arrays_hc)):
    #    if shifted_EEG.size > 0 and reduced_fnirs.size > 0:
    #        print(f"Shift {i + 1}:")
    #        print("Shifted EEG shape:", shifted_EEG.shape)
    #        print("Reduced fNIRS shape:", reduced_fnirs.shape)
    #        print("this is the length of shifted array", len(shifted_EEG_arrays_hc))
    #        print("this is the length of reduced fnirs array", len(reduced_fnirs_arrays_hc))
    #    else:
    #        print(f"Shift {i + 1}: Arrays are empty due to the shift amount.")

    hc_subj_shifted_EEG_arrays.append(shifted_EEG_arrays_hc)
    hc_subj_reduced_hbo_fnirs_arrays.append(reduced_hbo_fnirs_arrays_hc)
    hc_subj_reduced_hbr_fnirs_arrays.append(reduced_hbr_fnirs_arrays_hc)


# -----------------------------------------------------------------------------------------------------------------------------

p_subj_shifted_EEG_arrays = []
p_subj_reduced_hbo_fnirs_arrays = []
p_subj_reduced_hbr_fnirs_arrays = []

for subject_eeg_p, subject_hbo_fnirs_p, subject_hbr_fnirs_p in zip(subjects_eeg_p, fnirs_delays_hbo_p, fnirs_delays_hbr_p):
    shifted_EEG_arrays_p = []
    reduced_hbo_fnirs_arrays_p = []
    reduced_hbr_fnirs_arrays_p = []

    for shift in range(1, num_shifts + 1):
        shift_amount = int(shift * increment)
    
        # Shift EEG array to the right without filling with zeros by slicing
        if shift_amount < subject_eeg_p.shape[1]:
            shifted_EEG = subject_eeg_p[:, shift_amount:, :]
            
        else:
            shifted_EEG = np.array([])  # If shift_amount exceeds the array length, use an empty array

        # Reduce the corresponding portion from the fNIRS array
        reduced_hbo_fnirs = subject_hbo_fnirs_p[:-shift_amount, :, :]
        reduced_hbr_fnirs = subject_hbr_fnirs_p[:-shift_amount, :, :]

        # Ensure both arrays have compatible shapes
        if shifted_EEG.shape[1] == reduced_hbo_fnirs.shape[0]:
            shifted_EEG_arrays_p.append(shifted_EEG)
            reduced_hbo_fnirs_arrays_p.append(reduced_hbo_fnirs)
        else:
            # In case of dimension mismatch, adjust the lengths to match
            min_length = min(shifted_EEG.shape[1], reduced_hbo_fnirs.shape[0])
            if min_length > 0:
                shifted_EEG_arrays_p.append(shifted_EEG[:, :min_length, :])
                reduced_hbo_fnirs_arrays_p.append(reduced_hbo_fnirs[:min_length, :, :])
            else:
                # Handle the case where resulting arrays would be empty
                shifted_EEG_arrays_p.append(np.array([]))
                reduced_hbo_fnirs_arrays_p.append(np.array([]))
        
        # Ensure both arrays have compatible shapes
        if shifted_EEG.shape[1] == reduced_hbr_fnirs.shape[0]:
            shifted_EEG_arrays_p.append(shifted_EEG)
            reduced_hbr_fnirs_arrays_p.append(reduced_hbr_fnirs)
        else:
            # In case of dimension mismatch, adjust the lengths to match
            min_length = min(shifted_EEG.shape[1], reduced_hbr_fnirs.shape[0])
            if min_length > 0:
                shifted_EEG_arrays_p.append(shifted_EEG[:, :min_length, :])
                reduced_hbr_fnirs_arrays_p.append(reduced_hbr_fnirs[:min_length, :, :])
            else:
                # Handle the case where resulting arrays would be empty
                shifted_EEG_arrays_p.append(np.array([]))
                reduced_hbr_fnirs_arrays_p.append(np.array([]))
    
    # Example of how to access the shifted arrays and corresponding reduced arrays
    #for i, (shifted_EEG, reduced_fnirs) in enumerate(zip(shifted_EEG_arrays_p, reduced_fnirs_arrays_p)):
        #if shifted_EEG.size > 0 and reduced_fnirs.size > 0:
            #print(f"Shift {i + 1}:")
            #print("Shifted EEG shape:", shifted_EEG.shape)
            #print("Reduced fNIRS shape:", reduced_fnirs.shape)
            #print("this is the length of shifted array", len(shifted_EEG_arrays_p))
            #print("this is the length of reduced fnirs array", len(reduced_fnirs_arrays_p))
    #    else:
    #        print(f"Shift {i + 1}: Arrays are empty due to the shift amount.")

    p_subj_shifted_EEG_arrays.append(shifted_EEG_arrays_p)
    p_subj_reduced_hbo_fnirs_arrays.append(reduced_hbo_fnirs_arrays_p)
    p_subj_reduced_hbr_fnirs_arrays.append(reduced_hbr_fnirs_arrays_p)
# -----------------------------------------------------------------------------------------------------------------------------

from scipy.stats import pearsonr, spearmanr


def compute_and_plot_correlations(subjects_eeg, subjects_hbo_fnirs, subjects_hbr_fnirs, num_shifts, title_suffix):
    
    correlation_eeg_subj_theta = []
    correlation_eeg_subj_delta = []

    correlation_hbo_fnirs_subj = []
    correlation_hbr_fnirs_subj = []

    for subject_eeg in subjects_eeg:
        correlation_eeg_theta= []
        correlation_eeg_delta = []
        for ind in subject_eeg:
            if ind.ndim == 3:
                theta = ind[1, :, :]
                delta = ind[0,:,:]
                mean_chann_theta = np.mean(theta, axis=-1)
                mean_chann_delta = np.mean(delta, axis=-1)
                correlation_eeg_theta.append(mean_chann_theta)
                correlation_eeg_delta.append(mean_chann_delta)
        correlation_eeg_subj_theta.append(correlation_eeg_theta)
        correlation_eeg_subj_delta.append(correlation_eeg_delta)

    for subject_fnirs_hbo in subjects_hbo_fnirs:
        correlation_fnirs_hbo = []
        for delay_fnirs_hbo in subject_fnirs_hbo:
            if delay_fnirs_hbo.ndim == 3:
                mean_hbo = delay_fnirs_hbo[:, :, 0]
                mean_chann_hbo = np.mean(mean_hbo, axis=-1)
                correlation_fnirs_hbo.append(mean_chann_hbo)
        correlation_hbo_fnirs_subj.append(correlation_fnirs_hbo)
    
    for subject_fnirs_hbr in subjects_hbr_fnirs:
        correlation_fnirs_hbr = []
        for delay_fnirs_hbr in subject_fnirs_hbr:
            if delay_fnirs_hbr.ndim == 3:
                mean_hbr = delay_fnirs_hbr[:, :, 0]
                mean_chann_hbr = np.mean(mean_hbr, axis=-1)
                correlation_fnirs_hbr.append(mean_chann_hbr)
        correlation_hbr_fnirs_subj.append(correlation_fnirs_hbr)


    subj_correlation_results = []
    subj_pvalue_results = []
    for corr_eeg, corr_fnirs in zip(correlation_eeg_subj_theta, correlation_hbo_fnirs_subj):
        correlation_results = []
        pvalue_results = []
        for features_eeg, features_fnirs in zip(corr_eeg, corr_fnirs):
            correlation_coefficient, p_value = pearsonr(features_eeg, features_fnirs)
            correlation_results.append(correlation_coefficient)
            pvalue_results.append(p_value)
    
        subj_correlation_results.append(correlation_results)
        subj_pvalue_results.append(pvalue_results)

    max_corr_values = []
    index_max_corr = []
    min_pvalues = []
    index_min_pval = []

    for corr, pval in zip(subj_correlation_results, subj_pvalue_results): 
        
        max_corr = max(corr)
        max_corr_index = corr.index(max_corr)
        min_pval = min(pval)
        min_pval_index = pval.index(min_pval)

        max_corr_values.append(max_corr)
        min_pvalues.append(min_pval)
        index_max_corr.append(max_corr_index)
        index_min_pval.append(min_pval_index)

        time_array = np.linspace(0, 20, len(corr))
        plt.plot(time_array, corr, color="r", label="corr")
        plt.plot(time_array, pval, color="g", label="pvalue")
        plt.title("Pearson Correlation coeficients" + title_suffix)
        plt.xlabel("delay (s)")
        plt.legend()

    plt.show()

    # Create an Excel writer object
    output_file_path = f"data_ind_subj_{title_suffix}.xlsx"
    with pd.ExcelWriter(output_file_path, engine='xlsxwriter') as writer:
        # Iterate over subjects and write each to a separate sheet
        for idx_subj, (corr, pval) in enumerate(zip(subj_correlation_results, subj_pvalue_results)):
            # Collect data into a list of dictionaries
            data_list = []
            for idx, correlation in enumerate(corr):
                p_value = pval[idx]
                data_list.append({
                    "Delay Index": idx,
                    "Correlation Coefficient": correlation,
                    "P-value": p_value
                })

            # Convert the list of dictionaries to a DataFrame
            df = pd.DataFrame(data_list)
            
            # Write the DataFrame to a new sheet named after the subject index
            sheet_name = f"Subject_{title_suffix}_{idx_subj}"
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"Data has been exported to {output_file_path}")

    for index_delay, eeg_subject in zip(index_min_pval, correlation_eeg_subj_theta):
        print("This is index delay", index_delay)
        eeg_plot = eeg_subject[index_delay]
        # Create the x-axis
        x = range(len(eeg_plot))
        plt.scatter(x, eeg_plot)
        #plt.show()
    
    sil_score_list = []
    db_score_list = []

    for index_delay, eeg_subject_theta, eeg_subject_delta, fnirs_hbo_subject, fnirs_hbr_subject in zip(index_min_pval, correlation_eeg_subj_theta, correlation_eeg_subj_delta, correlation_hbo_fnirs_subj, correlation_hbr_fnirs_subj):
        eeg_theta =  eeg_subject_theta[index_delay]
        eeg_delta = eeg_subject_delta[index_delay]
        fnirs_hbo = fnirs_hbo_subject[index_delay]
        fnirs_hbr = fnirs_hbr_subject[index_delay]
        X = np.column_stack((eeg_theta, eeg_delta, fnirs_hbo, fnirs_hbr))

        # standardize data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        #Apply PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        # Tune Kmeans parameters 
        from sklearn.model_selection import GridSearchCV
        param_grid = {'n_clusters': range(1,5)}
        grid = GridSearchCV(KMeans(), param_grid, cv=5)
        grid.fit(X_pca)
        best_kmeans = grid.best_estimator_

        best_kmeans.fit(X_pca)
        cluster_labels = best_kmeans.labels_

        from sklearn.metrics import silhouette_score, davies_bouldin_score

        sil_score = silhouette_score(X_scaled, cluster_labels)
        db_score = davies_bouldin_score(X_scaled, cluster_labels)

        sil_score_list.append(sil_score)
        db_score_list.append(db_score)

        #print("Silhouette score : ", sil_score)
        #print("Davies-Bouldin index: ", db_score)

        # Plot the clusters
        plt.scatter(X_pca[:,0], X_pca[:,1], c=cluster_labels, cmap="viridis")
        plt.scatter(best_kmeans.cluster_centers_[:,0], best_kmeans.cluster_centers_[:,1], s=300, c="red", marker="x")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.title("Kmeans clustering")
        plt.show()



    print("this is the sil score list", sil_score_list)
    print("this is the db score list", db_score_list)

# Compute and plot correlations for healthy controls
compute_and_plot_correlations(hc_subj_shifted_EEG_arrays, hc_subj_reduced_hbo_fnirs_arrays, hc_subj_reduced_hbr_fnirs_arrays, num_shifts, "Healthy Controls")

# Compute and plot correlations for patients
compute_and_plot_correlations(p_subj_shifted_EEG_arrays, p_subj_reduced_hbo_fnirs_arrays, p_subj_reduced_hbr_fnirs_arrays, num_shifts, "Patients")