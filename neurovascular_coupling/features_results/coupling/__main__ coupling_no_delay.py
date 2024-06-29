import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
from scipy.integrate import simps
from scipy.signal import welch
from autoreject import get_rejection_threshold
from scipy.stats import ttest_1samp
import seaborn as sns
sns.set(font_scale=1.2)
from mne.time_frequency import tfr_multitaper, tfr_morlet
from matplotlib.colors import TwoSlopeNorm
from sklearn.preprocessing import StandardScaler

from scipy.stats import pearsonr, spearmanr

import statistics


#from basic.arrange_files import read_files
#from neurovascular_coupling.Hemo import HemoData
from arrange_files import read_files
from Hemo import HemoData


"""

Characterize data

"""

# Function to visualize dropped epochs
def visualize_dropped_epochs(epochs_0back, epochs_1back, epochs_2back, epochs_3back):

    dropped_epochs_indices = []

    # Step 4: Retrieve the indices of the dropped epochs
    #dropped_epochs_indices_baseline_0back = epochs_baseline_0back.drop_log
    #dropped_epochs_indices_baseline_1back = epochs_baseline_1back.drop_log
    #dropped_epochs_indices_baseline_2back = epochs_baseline_2back.drop_log
    #dropped_epochs_indices_baseline_3back = epochs_baseline_3back.drop_log

    dropped_epochs_indices_0back = epochs_0back.drop_log
    dropped_epochs_indices_1back = epochs_1back.drop_log
    dropped_epochs_indices_2back = epochs_2back.drop_log
    dropped_epochs_indices_3back = epochs_3back.drop_log

    # Convert the drop_log to a list of indices of dropped epochs
    #dropped_indices_baseline_0back = [i for i, log in enumerate(dropped_epochs_indices_baseline_0back) if len(log) > 0]
    #dropped_indices_baseline_1back = [i for i, log in enumerate(dropped_epochs_indices_baseline_1back) if len(log) > 0]
    #dropped_indices_baseline_2back = [i for i, log in enumerate(dropped_epochs_indices_baseline_2back) if len(log) > 0]
    #dropped_indices_baseline_3back = [i for i, log in enumerate(dropped_epochs_indices_baseline_3back) if len(log) > 0]

    dropped_indices_0back = [i for i, log in enumerate(dropped_epochs_indices_0back) if len(log) > 0]
    dropped_indices_1back = [i for i, log in enumerate(dropped_epochs_indices_1back) if len(log) > 0]
    dropped_indices_2back = [i for i, log in enumerate(dropped_epochs_indices_2back) if len(log) > 0]
    dropped_indices_3back = [i for i, log in enumerate(dropped_epochs_indices_3back) if len(log) > 0]

    #dropped_epochs_indices.append(dropped_indices_baseline_0back)
    dropped_epochs_indices.append(dropped_indices_0back)
    #dropped_epochs_indices.append(dropped_indices_baseline_1back)
    dropped_epochs_indices.append(dropped_indices_1back)
    #dropped_epochs_indices.append(dropped_indices_baseline_2back)
    dropped_epochs_indices.append(dropped_indices_2back)
    #dropped_epochs_indices.append(dropped_indices_baseline_3back)
    dropped_epochs_indices.append(dropped_indices_3back)
    

    return dropped_epochs_indices


# Function to characterize EEG data
def characterization_eeg(raw, epoch_duration):
    
    # Dictionary to store trigger data
    trigger_data = {
        '2': {'begin': [], 'end': [], 'duration': []},
        '4': {'begin': [], 'end': [], 'duration': []},
        '5': {'begin': [], 'end': [], 'duration': []},
        '6': {'begin': [], 'end': [], 'duration': []},
        '7': {'begin': [], 'end': [], 'duration': []}
    }
    annot = raw.annotations
    
    # Triggers 5, 6 & 7 --> 1-Back Test, 2-Back Test, 3-Back Test
    # Iterate over annotations 
    for idx, desc in enumerate(annot.description):
        if desc in trigger_data and desc != '2':
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
    
    for index, annotation in enumerate(annot.description):
        if (len(annot) == 11 and index==7):
            begin_relax_0back = annot.onset[7]
            duration_relax_0back = annot.duration[7] + 30
            end_relax_0back = begin_relax_0back + duration_relax_0back

            # Store trigger data in the dictionary
            trigger_data[annotation]["begin"].append(begin_relax_0back)
            trigger_data[annotation]["end"].append(end_relax_0back)
            trigger_data[annotation]["duration"].append(duration_relax_0back)

        elif (len(annot) == 9 and index==5):
            begin_relax_0back = annot.onset[5]
            duration_relax_0back = annot.duration[5] + 30
            end_relax_0back = begin_relax_0back + duration_relax_0back

            # Store trigger data in the dictionary
            trigger_data[annotation]["begin"].append(begin_relax_0back)
            trigger_data[annotation]["end"].append(end_relax_0back)
            trigger_data[annotation]["duration"].append(duration_relax_0back)

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
    # consider that the instruction period is 10 seconds 

    raw_baseline_0back = raw_new.copy().crop(tmin=trigger_data['2']['begin'][0] + 15, tmax=trigger_data['2']['end'][0])
    raw_0back = raw_new.copy().crop(tmin=trigger_data['4']['begin'][0], tmax=trigger_data['4']['end'][0])
    raw_baseline_1back = raw_new.copy().crop(tmin=trigger_data['4']['end'][0], tmax=trigger_data['4']['end'][0] + 15)
    raw_1back = raw_new.copy().crop(tmin=trigger_data['5']['begin'][0], tmax=trigger_data['5']['end'][0])
    raw_baseline_2back = raw_new.copy().crop(tmin=trigger_data['5']['end'][0], tmax=trigger_data['5']['end'][0] + 15)
    raw_2back = raw_new.copy().crop(tmin=trigger_data['6']['begin'][0], tmax=trigger_data['6']['end'][0])
    raw_baseline_3back = raw_new.copy().crop(tmin=trigger_data['6']['end'][0], tmax=trigger_data['6']['end'][0] + 15)
    raw_3back = raw_new.copy().crop(tmin=trigger_data['7']['begin'][0], tmax=trigger_data['7']['end'][0])

    # Epoch data

    """
    Returns a mne.Epochs object created from the annotated mne.io.Raw object.

    """

    eeg_epochs = list()
    eeg_baseline_epochs = list()

    eeg_bad_epochs = list()

    baseline_0back = mne.make_fixed_length_events(raw_baseline_0back, start = 0 , stop = raw_baseline_0back.times[-1], duration = epoch_duration)
    baseline_1back = mne.make_fixed_length_events(raw_baseline_1back, start = 0 , stop = raw_baseline_1back.times[-1], duration = epoch_duration)
    baseline_2back = mne.make_fixed_length_events(raw_baseline_2back, start = 0 , stop = raw_baseline_2back.times[-1], duration = epoch_duration)
    baseline_3back = mne.make_fixed_length_events(raw_baseline_3back, start = 0 , stop = raw_baseline_3back.times[-1], duration = epoch_duration)

    events_0back = mne.make_fixed_length_events(raw_0back, start = 0 , stop = raw_0back.times[-1], duration = epoch_duration)
    events_1back = mne.make_fixed_length_events(raw_1back, start = 0 , stop = raw_1back.times[-1], duration = epoch_duration)
    events_2back = mne.make_fixed_length_events(raw_2back, start = 0 , stop = raw_2back.times[-1], duration = epoch_duration)
    events_3back = mne.make_fixed_length_events(raw_3back, start = 0 , stop = raw_3back.times[-1], duration = epoch_duration)
    
    epochs_baseline_0back = mne.Epochs(raw_baseline_0back, baseline_0back, tmin=0, tmax= epoch_duration, baseline = None, preload = True)
    epochs_baseline_1back = mne.Epochs(raw_baseline_1back, baseline_1back, tmin=0, tmax= epoch_duration, baseline = None, preload = True)
    epochs_baseline_2back = mne.Epochs(raw_baseline_2back, baseline_2back, tmin=0, tmax= epoch_duration, baseline = None, preload = True)
    epochs_baseline_3back = mne.Epochs(raw_baseline_3back, baseline_3back, tmin=0, tmax= epoch_duration, baseline = None, preload = True)

    epochs_0back = mne.Epochs(raw_0back, events_0back, tmin=0, tmax= epoch_duration, baseline = None, preload = True)
    epochs_1back = mne.Epochs(raw_1back, events_1back, tmin=0, tmax= epoch_duration, baseline = None, preload = True)
    epochs_2back = mne.Epochs(raw_2back, events_2back, tmin=0, tmax= epoch_duration, baseline = None, preload = True)
    epochs_3back = mne.Epochs(raw_3back, events_3back, tmin=0, tmax= epoch_duration, baseline = None, preload = True)
        
    # -----------------Get-Rejection-Threshold---------------------------------
    
    #reject_thresholds_baseline_0back = get_rejection_threshold(epochs_baseline_0back, ch_types = "eeg", verbose = False)
    #reject_thresholds_baseline_1back = get_rejection_threshold(epochs_baseline_1back, ch_types = "eeg", verbose = False)
    #reject_thresholds_baseline_2back = get_rejection_threshold(epochs_baseline_2back, ch_types = "eeg", verbose = False)
    #reject_thresholds_baseline_3back = get_rejection_threshold(epochs_baseline_3back, ch_types = "eeg", verbose = False)
    
    #reject_thresholds_0back = get_rejection_threshold(epochs_0back, ch_types = "eeg", verbose = False)
    #reject_thresholds_1back = get_rejection_threshold(epochs_1back, ch_types = "eeg", verbose = False)
    #reject_thresholds_2back = get_rejection_threshold(epochs_2back, ch_types = "eeg", verbose = False)
    #reject_thresholds_3back = get_rejection_threshold(epochs_3back, ch_types = "eeg", verbose = False)
        
    #epochs_baseline_0back.drop_bad(reject=reject_thresholds_baseline_0back)
    #epochs_baseline_1back.drop_bad(reject=reject_thresholds_baseline_1back)
    #epochs_baseline_2back.drop_bad(reject=reject_thresholds_baseline_2back)
    #epochs_baseline_3back.drop_bad(reject=reject_thresholds_baseline_3back)

    #epochs_0back.drop_bad(reject=reject_thresholds_0back)
    #epochs_1back.drop_bad(reject=reject_thresholds_1back)
    #epochs_2back.drop_bad(reject=reject_thresholds_2back)
    #epochs_3back.drop_bad(reject=reject_thresholds_3back)

    # Visualize dropped epochs
    #bad_epoch = visualize_dropped_epochs(epochs_0back, epochs_1back, epochs_2back, epochs_3back, epochs_baseline_0back, epochs_baseline_1back, epochs_baseline_2back, epochs_baseline_3back)
    bad_epoch = visualize_dropped_epochs(epochs_0back, epochs_1back, epochs_2back, epochs_3back)

    eeg_epochs.append(epochs_0back)
    eeg_epochs.append(epochs_1back)
    eeg_epochs.append(epochs_2back)
    eeg_epochs.append(epochs_3back)

    eeg_baseline_epochs.append(epochs_baseline_0back)
    eeg_baseline_epochs.append(epochs_baseline_1back)
    eeg_baseline_epochs.append(epochs_baseline_2back)
    eeg_baseline_epochs.append(epochs_baseline_3back)

    eeg_bad_epochs.append(bad_epoch)

    return (eeg_epochs, eeg_baseline_epochs, eeg_bad_epochs)

def characterization_fNIRS(raw, epoch_duration, bad_epochs):
    
    # Dictionary to store trigger data
    trigger_data = {
        '2': {'begin': [], 'end': [], 'duration': []},
        '4': {'begin': [], 'end': [], 'duration': []},
        '5': {'begin': [], 'end': [], 'duration': []},
        '6': {'begin': [], 'end': [], 'duration': []},
        '7': {'begin': [], 'end': [], 'duration': []}
    }

    annot = raw.annotations

    # Triggers 5, 6 & 7 --> 1-Back Test, 2-Back Test, 3-Back Test
    # Iterate over annotations 
    for idx, desc in enumerate(annot.description):
        if desc in trigger_data and desc != '2':
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


    for index, annotation in enumerate(annot.description):
        if (len(annot) == 11 and index==7):
            begin_relax_0back = annot.onset[7]
            duration_relax_0back = annot.duration[7] + 20
            end_relax_0back = begin_relax_0back + duration_relax_0back

            # Store trigger data in the dictionary
            trigger_data[annotation]["begin"].append(begin_relax_0back)
            trigger_data[annotation]["end"].append(end_relax_0back)
            trigger_data[annotation]["duration"].append(duration_relax_0back)
        elif (len(annot) == 9 and index==5):
            begin_relax_0back = annot.onset[5]
            duration_relax_0back = annot.duration[5] + 20
            end_relax_0back = begin_relax_0back + duration_relax_0back

            # Store trigger data in the dictionary
            trigger_data[annotation]["begin"].append(begin_relax_0back)
            trigger_data[annotation]["end"].append(end_relax_0back)
            trigger_data[annotation]["duration"].append(duration_relax_0back)

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
    # we will consider that the instructions last 15 seconds 
      
    raw_baseline_0back = raw_new.copy().crop(tmin=trigger_data['2']['begin'][0] + 15, tmax=trigger_data['2']['end'][0])
    raw_0back = raw_new.copy().crop(tmin=trigger_data['4']['begin'][0], tmax=trigger_data['4']['end'][0])
    raw_baseline_1back = raw_new.copy().crop(tmin=trigger_data['4']['end'][0], tmax=trigger_data['4']['end'][0] + 15)
    raw_1back = raw_new.copy().crop(tmin=trigger_data['5']['begin'][0], tmax=trigger_data['5']['end'][0])
    raw_baseline_2back = raw_new.copy().crop(tmin=trigger_data['5']['end'][0], tmax=trigger_data['5']['end'][0] + 15)
    raw_2back = raw_new.copy().crop(tmin=trigger_data['6']['begin'][0], tmax=trigger_data['6']['end'][0])
    raw_baseline_3back = raw_new.copy().crop(tmin=trigger_data['6']['end'][0], tmax=trigger_data['6']['end'][0] + 15)
    raw_3back = raw_new.copy().crop(tmin=trigger_data['7']['begin'][0], tmax=trigger_data['7']['end'][0])

    # Epoch data

    fnirs_epochs = list()
    fnirs_baseline_epochs = list()

    for bad in bad_epochs:
        #print("This is bad", bad)

        baseline_0back = mne.make_fixed_length_events(raw_baseline_0back, start = 0 , stop = raw_baseline_0back.times[-1], duration = epoch_duration)
        baseline_1back = mne.make_fixed_length_events(raw_baseline_1back, start = 0 , stop = raw_baseline_1back.times[-1], duration = epoch_duration)
        baseline_2back = mne.make_fixed_length_events(raw_baseline_2back, start = 0 , stop = raw_baseline_2back.times[-1], duration = epoch_duration)
        baseline_3back = mne.make_fixed_length_events(raw_baseline_3back, start = 0 , stop = raw_baseline_3back.times[-1], duration = epoch_duration)

        events_0back = mne.make_fixed_length_events(raw_0back, start = 0 , stop = raw_0back.times[-1], duration = epoch_duration)
        events_1back = mne.make_fixed_length_events(raw_1back, start = 0 , stop = raw_1back.times[-1], duration = epoch_duration)
        events_2back = mne.make_fixed_length_events(raw_2back, start = 0 , stop = raw_2back.times[-1], duration = epoch_duration)
        events_3back = mne.make_fixed_length_events(raw_3back, start = 0 , stop = raw_3back.times[-1], duration = epoch_duration)
        
        epochs_baseline_0back = mne.Epochs(raw_baseline_0back, baseline_0back, tmin=0, tmax= epoch_duration, baseline = None, preload = True)
        epochs_baseline_1back = mne.Epochs(raw_baseline_1back, baseline_1back, tmin=0, tmax= epoch_duration, baseline = None, preload = True)
        epochs_baseline_2back = mne.Epochs(raw_baseline_2back, baseline_2back, tmin=0, tmax= epoch_duration, baseline = None, preload = True)
        epochs_baseline_3back = mne.Epochs(raw_baseline_3back, baseline_3back, tmin=0, tmax= epoch_duration, baseline = None, preload = True)

        epochs_0back = mne.Epochs(raw_0back, events_0back, tmin=0, tmax= epoch_duration, baseline = None, preload = True)
        epochs_1back = mne.Epochs(raw_1back, events_1back, tmin=0, tmax= epoch_duration, baseline = None, preload = True)
        epochs_2back = mne.Epochs(raw_2back, events_2back, tmin=0, tmax= epoch_duration, baseline = None, preload = True)
        epochs_3back = mne.Epochs(raw_3back, events_3back, tmin=0, tmax= epoch_duration, baseline = None, preload = True)

        #epochs_baseline_0back.drop(bad[0][0:-1])
        #epochs_0back.drop(bad[0][0:-1])
        #epochs_baseline_1back.drop(bad[2][0:-1])
        #epochs_1back.drop(bad[1][0:-1])
        #epochs_baseline_2back.drop(bad[4][0:-1])
        #epochs_2back.drop(bad[2][0:-1])
        #epochs_baseline_3back.drop(bad[6][0:-1])
        #epochs_3back.drop(bad[3][0:-1])

        fnirs_baseline_epochs.append(epochs_baseline_0back)
        fnirs_baseline_epochs.append(epochs_baseline_1back)
        fnirs_baseline_epochs.append(epochs_baseline_2back)
        fnirs_baseline_epochs.append(epochs_baseline_3back)

        fnirs_epochs.append(epochs_0back)
        fnirs_epochs.append(epochs_1back)
        fnirs_epochs.append(epochs_2back)
        fnirs_epochs.append(epochs_3back)

    return (fnirs_epochs, fnirs_baseline_epochs)

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

def psd_values_from_bands(data, sfreq, bands, tmin=0, tmax=None, ln_normalization=False, method='multitaper'):
    """
    Compute PSD values for each frequency band using MNE.

    Parameters:
    - data: array of shape (epochs, channels, times)
    - sfreq: sampling frequency
    - bands: list of tuples, each containing (fmin, fmax, label)
    - tmin: start time for PSD calculation
    - tmax: end time for PSD calculation
    - ln_normalization: boolean, whether to apply logarithmic normalization
    - method: 'multitaper' or 'welch', method for PSD calculation

    Returns:
    - bp: array of shape (number of bands, epochs, channels, frequencies in the band)
    """
    if method not in ['multitaper', 'welch']:
        raise ValueError("method should be either 'multitaper' or 'welch'")

    n_epochs, n_channels, _ = data.shape
    
    # Compute the PSD using MNE
    if method == 'multitaper':
        psds, freqs = mne.time_frequency.psd_array_multitaper(data, sfreq=sfreq, verbose=False)
    elif method == 'welch':
        psds, freqs = mne.time_frequency.psd_array_welch(data, sfreq=sfreq, verbose=False)
    
    # Initialize the output array
    band_psd_list = []

    for band in bands:
        b0, b1, _ = band
        idx_band = np.logical_and(freqs >= b0, freqs <= b1)
        band_psd = psds[:, :, idx_band]
        
        if ln_normalization:
            band_psd = np.log(band_psd)
        
        band_psd_list.append(band_psd)
    
    # Determine the maximum number of frequencies in any band
    max_freqs = max(band.shape[-1] for band in band_psd_list)
    num_bands = len(bands)

    # Prepare an array to store the band PSD values with shape (num_bands, num_epochs, num_channels, max_freqs)
    bp = np.zeros((num_bands, n_epochs, n_channels, max_freqs), dtype=np.float64)
    
    for i, band_psd in enumerate(band_psd_list):
        bp[i, ..., :band_psd.shape[-1]] = band_psd
    
    print("this is the shape of bp", bp.shape)

    return bp

"""

Process data

"""

# Function to extract the mean HBO concentration from fNIRS epochs
def extract_fnirs_features(epochs_fnirs):
    features_fnirs = []
    for epoch in epochs_fnirs:
        raw_hbo = epoch.get_data(picks=["hbo"])
        ss = StandardScaler()

        # Apply StandardScaler independently to each slice along the first dimension (axis=0)
        for i in range(raw_hbo.shape[0]):
            raw_hbo[i] = ss.fit_transform(raw_hbo[i])
        
        mean_std_per_epoch_per_channel = np.concatenate([
            np.expand_dims(np.mean(raw_hbo, axis=-1), axis=-1),
            np.expand_dims(np.std(raw_hbo, axis=-1), axis=-1)], axis=-1)
        features_fnirs.append(mean_std_per_epoch_per_channel)
    return features_fnirs

# Function to subtract baseline from fNIRS features
def subtract_baseline_fnirs(nback_features, baseline_features):
    
    corrected_features = []
    
    for i in range(len(nback_features)):
        
        # Extract the HbO channels from the baseline features
        baseline_mean_hbo = baseline_features[i][:, [2, 3, 6, 7], 0]
        
        # Calculate the mean of the baseline HbO epochs
        baseline_mean = np.mean(baseline_mean_hbo, axis=0, keepdims=True)
        
        # Extract the HbO channels from the nback features
        nback_features_hbo = nback_features[i][:, [2, 3, 6, 7], 0]
        
        # Subtract the baseline mean from the nback HbO channels
        nback_features_hbo_corrected = nback_features_hbo - baseline_mean
        
        # Append the corrected HbO features to the list
        corrected_features.append(nback_features_hbo_corrected)
    
    return corrected_features

# Function to process fNIRS data
def process_fnirs_data(file_dirs_fnirs, epoch_duration, bad_epochs):
    results = { "nback": [[], [], [], []], "baseline": [[], [], [], []] }

    for file_dir, bad_epoch in zip(file_dirs_fnirs, bad_epochs):
        raw_haemo = HemoData(file_dir, preprocessing=True, isPloting=False).getMneIoRaw()

        #raw_haemo.plot(show_options = True, block=True)
        #plt.show()

        epochs_fnirs = characterization_fNIRS(raw_haemo, epoch_duration, bad_epoch)

        nback_epochs = epochs_fnirs[0]
        baseline_epochs = epochs_fnirs[1]

        # Extract features
        features_nback = extract_fnirs_features(nback_epochs)
        features_baseline = extract_fnirs_features(baseline_epochs)

        # Subtract baseline from each n-back task
        features_nback = subtract_baseline_fnirs(features_nback, features_baseline)

        for i in range(len(nback_epochs)):
            results["nback"][i].append(features_nback[i])
            results["baseline"][i].append(features_baseline[i])

    return results

# Function to subtract baseline from EEG bandpower
def subtract_baseline_eeg(nback_bandpower, baseline_bandpower):
    theta_power_corrected = []
    
    for i in range(len(nback_bandpower)):
        
        baseline_mean_theta = baseline_bandpower[i][1, :, :]
        baseline_mean = np.mean(baseline_mean_theta, axis=0, keepdims=True)
        
        nback_bandpower_theta = nback_bandpower[i][1, :, :]
        nback_bandpower_theta_corrected = nback_bandpower_theta - baseline_mean
        
        theta_power_corrected.append(nback_bandpower_theta_corrected)

    return theta_power_corrected

# Function to extract the Relative Bandpower
def extract_bandpower_from_epochs(epochs, sf, bands, plot=False):
    bp_relative = []

    for nback in epochs:
        data = nback.get_data(units="uV")
        sf = nback.info['sfreq']

        nperseg = 2 * sf
        noverlap = int(0.75 * nperseg)
        freqs, psd = welch(data, sf, nperseg=nperseg, noverlap=noverlap, average="median")

        if plot:
            psd_mean = np.mean(psd, axis=0)

            # Plot the power spectrum 
            sns.set_theme(font_scale=1.2, style="white")
            plt.figure(figsize=(8, 4))
            plt.plot(freqs, psd_mean[0, :], color="k", lw=2)
            plt.fill_between(freqs, psd_mean[0, :], color='skyblue', alpha=0.4)
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('PSD log($\mu V^2$/Hz)')
            plt.yscale('log')
            plt.ylim([0, psd_mean[0, :].max() * 1.1])
            plt.xlim([0, freqs.max()])
            plt.xlim([0, 50])
            sns.despine()
            plt.show()

        bp = bandpower_from_psd_ndarray(psd, freqs, bands, ln_normalization=False, relative=True)
        #bp = psd_values_from_bands(data, sf, bands, ln_normalization=True, method='welch')
        bp_relative.append(bp)

    return bp_relative, freqs

# Function to process EEG data
def process_eeg_data(file_dirs, epoch_duration, bands):
    results = {"nback": [[], [], [], []], "baseline": [[], [], [], []]}
    
    subj_bad_epochs = []

    for file in file_dirs:
        raw = mne.io.read_raw_fif(file, preload=True)

        # Channels to drop
        channels_to_drop = ["AF7", "AFF5h", "AF8", "AFF6h", "FT7", "FC5", "FC3", "FCC3h", "CCP3h",
                            "CP1", "CPP3h", "P1", "TP7", "CCP1h", "CCP2h", "CP2", "CPP4h", "P2", "TP8",
                            "CCP4h", "FCC4h", "FC4", "FC6", "FT8",  "FCC1h", "FCC2h"]

        # Drop specified channels
        raw.drop_channels(channels_to_drop)

        raw.load_data()

        epochs, baseline_epochs, bad_epochs = characterization_eeg(raw, epoch_duration)
        sf = raw.info['sfreq']
        
        # Extract bandpower
        bp_nback, freqs = extract_bandpower_from_epochs(epochs, sf, bands, plot=False)
        bp_baseline, _ = extract_bandpower_from_epochs(baseline_epochs, sf, bands, plot=False)

        # Subtract baseline from each n-back task
        bp_nback = subtract_baseline_eeg(bp_nback, bp_baseline)

        for i, (bp_n, bp_b) in enumerate(zip(bp_nback, bp_baseline)):
            results["nback"][i].append(bp_n)
            results["baseline"][i].append(bp_b)

        subj_bad_epochs.append(bad_epochs)

    return results, subj_bad_epochs

def coupling(data_eeg, data_fnirs):
    # Collect mean_channels from each array
    subjects_eeg_list = []
    subjects_fnirs_list = []

    for (array_eeg, array_fnirs) in zip(data_eeg, data_fnirs):
        mean_channels_eeg = np.mean(array_eeg, axis=-1)
        mean_channels_fnirs = np.mean(array_fnirs, axis=-1)

        normalized_eeg = (mean_channels_eeg - np.mean(mean_channels_eeg))/ np.std(mean_channels_eeg)
        normalized_fnirs = (mean_channels_fnirs - np.mean(mean_channels_fnirs))/ np.std(mean_channels_fnirs)

        subjects_eeg_list.append(normalized_eeg)
        subjects_fnirs_list.append(normalized_fnirs)
    
    return subjects_eeg_list, subjects_fnirs_list

def correlation(theta_power, hbo_concentration):
    
    correlation_coefficient, p_value = pearsonr(theta_power, hbo_concentration)
    
    print("This is the correlation coefficient", correlation_coefficient)
    print("This is the pvalue", p_value)

if __name__ == "__main__":
    def main():
        os.chdir("/Users/adriana/Documents/GitHub/thesis")
        #os.chdir("H:\Dokumenter\GitHub\MasterThesis\.venv")
        mne.set_log_level('error')
        mne.set_config('MNE_BROWSER_BACKEND', 'matplotlib')

        clean_raw_eeg_hc = "/Users/adriana/Documents/DTU/thesis/data_acquisition/clean_eeg/healthy_controls/"
        clean_raw_eeg_p = "/Users/adriana/Documents/DTU/thesis/data_acquisition/clean_eeg/patients/"
        clean_raw_fnirs_hc = "/Users/adriana/Documents/DTU/thesis/data_acquisition/data_fnirs/healthy_controls/"
        clean_raw_fnirs_p = "/Users/adriana/Documents/DTU/thesis/data_acquisition/data_fnirs/patients/"

        #clean_raw_eeg_hc = "H:\\Dokumenter\\data_acquisition\\data_eeg\\clean_eeg\\healthy_controls\\try\\"
        #clean_raw_eeg_p = "H:\\Dokumenter\\data_acquisition\\data_eeg\\clean_eeg\\patients\\try\\"
        #clean_raw_fnirs_hc = "H:\\Dokumenter\\data_acquisition\\data_fnirs\\healthy_controls\\baseline\\snirf_files\\try\\"
        #clean_raw_fnirs_p = "H:\\Dokumenter\\data_acquisition\\data_fnirs\\patients\\baseline\\snirf_files\\try\\"

        file_dirs_eeg_hc, _ = read_files(clean_raw_eeg_hc, '.fif')
        file_dirs_eeg_p, _ = read_files(clean_raw_eeg_p, '.fif')
        file_dirs_fnirs_hc, _ = read_files(clean_raw_fnirs_hc, '.snirf')
        file_dirs_fnirs_p, _ = read_files(clean_raw_fnirs_p, '.snirf')

        bands = [(0.5, 4, 'Delta'), (4, 8, 'Theta'), (8, 12, 'Alpha'), 
                 (12, 16, 'Sigma'), (16, 30, 'Beta')]
        
        epoch_duration = 3

        # Process EEG data
        eeg_hc_results, subj_hc_bad_epochs = process_eeg_data(file_dirs_eeg_hc, epoch_duration, bands)
        eeg_p_results, subj_p_bad_epochs = process_eeg_data(file_dirs_eeg_p, epoch_duration, bands)

        # Process fNIRS data
        fnirs_hc_results = process_fnirs_data(file_dirs_fnirs_hc, epoch_duration, subj_hc_bad_epochs)
        fnirs_p_results = process_fnirs_data(file_dirs_fnirs_p, epoch_duration, subj_p_bad_epochs)
        
        nback_hc_theta = []
        nback_p_theta = []

        nback_hc_hbo = []
        nback_p_hbo = []
        
        for i in range(4):
            hc_subjects_theta, hc_subjects_hbo = coupling(eeg_hc_results["nback"][i], fnirs_hc_results["nback"][i])
            p_subjects_theta, p_subjects_hbo =  coupling(eeg_p_results["nback"][i], fnirs_p_results["nback"][i])

            nback_hc_theta.append(hc_subjects_theta)
            nback_p_theta.append(p_subjects_theta)

            nback_hc_hbo.append(hc_subjects_hbo)
            nback_p_hbo.append(p_subjects_hbo)
        
        for nback in nback_hc_theta:
            for subject in nback:
                print(subject.shape)
        
        print("this is the length of subjects", len(hc_subjects_theta))
                
        # Number of iterations to concatenate arrays
        iterations_hc = len(hc_subjects_theta)
        interval_hc = len(hc_subjects_theta)

        iterations_p = len(p_subjects_theta)
        interval_p = len(p_subjects_theta)
        
        # Flatten the list of arrays
        flattened_nback_hc_theta = [array for sublist in nback_hc_theta for array in sublist]
        flattened_nback_p_theta = [array for sublist in nback_p_theta for array in sublist]
        flattened_nback_hc_hbo = [array for sublist in nback_hc_hbo for array in sublist]
        flattened_nback_p_hbo = [array for sublist in nback_p_hbo for array in sublist]

        # Concatenate arrays based on the interval
        result_hc_theta = []
        result_p_theta = []
        result_hc_hbo = []
        result_p_hbo = []

        for start_hc in range(iterations_hc):
            concat_array_hc_theta = np.concatenate(flattened_nback_hc_theta[start_hc::interval_hc])
            concat_array_hc_hbo = np.concatenate(flattened_nback_hc_hbo[start_hc::interval_hc])

            result_hc_theta.append(concat_array_hc_theta)
            result_hc_hbo.append(concat_array_hc_hbo)
        
        for start_p in range(iterations_p):
            concat_array_p_theta = np.concatenate(flattened_nback_p_theta[start_p::interval_p])
            concat_array_p_hbo = np.concatenate(flattened_nback_p_hbo[start_p::interval_p])

            result_p_theta.append(concat_array_p_theta)
            result_p_hbo.append(concat_array_p_hbo)

        result_hc_theta = np.concatenate(result_hc_theta, axis=0)
        result_p_theta = np.concatenate(result_p_theta, axis=0)
        result_hc_hbo = np.concatenate(result_hc_hbo, axis=0)
        result_p_hbo = np.concatenate(result_p_hbo, axis=0)

        correlation(result_hc_theta, result_hc_hbo)
        correlation(result_p_theta, result_p_hbo)

    main()