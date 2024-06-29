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
from matplotlib.patches import Patch

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
    
    reject_thresholds_0back = get_rejection_threshold(epochs_0back, ch_types = "eeg", verbose = False)
    reject_thresholds_1back = get_rejection_threshold(epochs_1back, ch_types = "eeg", verbose = False)
    reject_thresholds_2back = get_rejection_threshold(epochs_2back, ch_types = "eeg", verbose = False)
    reject_thresholds_3back = get_rejection_threshold(epochs_3back, ch_types = "eeg", verbose = False)
        
    #epochs_baseline_0back.drop_bad(reject=reject_thresholds_baseline_0back)
    #epochs_baseline_1back.drop_bad(reject=reject_thresholds_baseline_1back)
    #epochs_baseline_2back.drop_bad(reject=reject_thresholds_baseline_2back)
    #epochs_baseline_3back.drop_bad(reject=reject_thresholds_baseline_3back)

    epochs_0back.drop_bad(reject=reject_thresholds_0back)
    epochs_1back.drop_bad(reject=reject_thresholds_1back)
    epochs_2back.drop_bad(reject=reject_thresholds_2back)
    epochs_3back.drop_bad(reject=reject_thresholds_3back)

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
        
        #band_psd = psd[..., idx_band]
        #bp[i] = np.mean(band_psd, axis=-1)

    if relative:
        bp /= total_power

    if ln_normalization:
        bp = 10*np.log10(bp)
    
    print("this is the shape of bp", bp.shape)

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
        #baseline_mean_hbo = baseline_features[i][:, [2, 3, 6, 7], 0]
        baseline_mean_hbo = baseline_features[i][:, :, 0]
        
        # Calculate the mean of the baseline HbO epochs
        baseline_mean = np.mean(baseline_mean_hbo, axis=0, keepdims=True)
        
        # Extract the HbO channels from the nback features
        #nback_features_hbo = nback_features[i][:, [2, 3, 6, 7], 0]
        nback_features_hbo = nback_features[i][:, :, 0]
        
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

    return raw_haemo, results

# Function to subtract baseline from EEG bandpower
def subtract_baseline_eeg(nback_bandpower, baseline_bandpower):
    theta_power_corrected = []
    
    for i in range(len(nback_bandpower)):
        
        baseline_mean_theta = baseline_bandpower[i][1, :, :]
        baseline_mean = np.mean(baseline_mean_theta, axis=0, keepdims=True)
        
        nback_bandpower_theta = nback_bandpower[i][1, :, :]
        nback_bandpower_theta_corrected = nback_bandpower_theta - baseline_mean

        print("this is the shape of nback_bandpower_corrected", nback_bandpower_theta_corrected.shape)
        
        theta_power_corrected.append(nback_bandpower_theta_corrected)

    return theta_power_corrected

# Function to extract the Relative Bandpower
def extract_bandpower_from_epochs(epochs, sf, bands, plot=False):
    bp_relative = []

    for nback in epochs:
        data = nback.get_data(units="uV")

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

        bp = bandpower_from_psd_ndarray(psd, freqs, bands, ln_normalization=True, relative=False)
        #bp = psd_values_from_bands(data, sf, bands, ln_normalization=True, method='welch')
        bp_relative.append(bp)

    return bp_relative, freqs

# Function to process EEG data
def process_eeg_data(file_dirs, epoch_duration, bands):
    results = {"nback": [[], [], [], []], "baseline": [[], [], [], []]}
    
    subj_bad_epochs = []

    for file in file_dirs:
        raw = mne.io.read_raw_fif(file, preload=True)

        #raw.plot(show_options=True, block=True)
        #plt.show()

        raw.compute_psd(fmin=1.0, fmax=60.0).plot(
        average=True, amplitude=False, picks="data", exclude="bads"
        )

        # Channels to drop
        channels_to_drop = ["AF7", "AFF5h", "AF8", "AFF6h", "FT7", "FC5", "FC3", "FCC3h", "CCP3h",
                            "CP1", "CPP3h", "P1", "TP7", "CCP1h", "CCP2h", "CP2", "CPP4h", "P2", "TP8",
                            "CCP4h", "FCC4h", "FC4", "FC6", "FT8",  "FCC1h", "FCC2h"]

        # Drop specified channels
        raw.drop_channels(channels_to_drop)

        raw.set_montage("brainproducts-RNP-BA-128")
        raw.plot_sensors(show_names=True)
        plt.show()

        # Apply a notch filter 
        #raw.filter(l_freq=1, h_freq=35, verbose=False)

        raw.load_data()

        epochs, baseline_epochs, bad_epochs = characterization_eeg(raw, epoch_duration)
        sf = raw.info['sfreq']

        # Define frequency range and number of cycles for 0 to 15 Hz
        #freqs = np.logspace(*np.log10([0.5, 35]), num=8)
        #n_cycles = freqs / 2.0  # different number of cycle per frequency

        #freqs = np.linspace(1, 15, 50)
        #n_cycles = freqs / 2  # Different number of cycles per frequency

        # Compute time-frequency representation for the full range
        #tfr = plot_tfr(epochs, freqs, n_cycles)
        
        #tfr.plot([0], baseline=(None, 0), mode='logratio', title='TFR', show=False)
        #plt.show()
        
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

# Apply one sample t test to EEG data
def one_sample_ttest_eeg(data):
    
    # Collect mean_channels from each array
    subjects_eeg_list = []

    for array in data:
        mean_channels = np.mean(array, axis=-1)
        mean_epochs = np.mean(mean_channels, axis=0)
        subjects_eeg_list.append(mean_epochs)
    
    array_subj = np.array(subjects_eeg_list)
    print("this is the shape of array_subj", array_subj.shape)

    t_stat, p_value = ttest_1samp(np.array(subjects_eeg_list), popmean=0, axis=0)

    return subjects_eeg_list, t_stat, p_value

# Apply one sample t test to fNIRS data
def one_sample_ttest_fnirs(data):

    subjects_fnirs_list = []

    for array in data:
        mean_channels = np.mean(array, axis=-1)
        mean_epochs = np.mean(mean_channels, axis=0)
        subjects_fnirs_list.append(mean_epochs)

    t_stat, p_value = ttest_1samp(np.array(subjects_fnirs_list), popmean=0, axis=0)

    return subjects_fnirs_list, t_stat, p_value

def extract_bandpower_data(subjects_eeg):
    band_powers = {'theta': []}
    
    for subject in subjects_eeg:
        band_powers['theta'].append(np.mean(subject[:, :], axis=0))
    
    return band_powers

def plot_topomaps_avg(band_data, task, info, axes, vmin, vmax):
    avg_data = np.mean(band_data, axis=0)
    evoked = mne.EvokedArray(avg_data[:, np.newaxis], info)
    im, _ = mne.viz.plot_topomap(
        evoked.data[:, 0], evoked.info, extrapolate="box", cmap='Spectral_r',
        res=32, contours=4, show=False, axes=axes, vlim=(vmin, vmax)
    )
    axes.set_title(f'{task} Task')
    return im

def eeg_features(subjects_eeg_0back, subjects_eeg_1back, subjects_eeg_2back, subjects_eeg_3back, title_suffix):
    nback_tasks = {
        '0-Back': subjects_eeg_0back,
        '1-Back': subjects_eeg_1back,
        '2-Back': subjects_eeg_2back,
        '3-Back': subjects_eeg_3back
    }
    
    #channel_names = ["AFp1", "AFF1h", "FFC1h", "AFp2", "AFF2h", "FFC2h",
    #                "AF7", "AFF5h", "AF8", "AFF6h", "FT7", "FC5", "FC3", "FCC3h", "CCP3h",
    #                "CP1", "CPP3h", "P1", "TP7", "CCP1h", "CCP2h", "CP2", "CPP4h", "P2", "TP8",
    #                "CCP4h", "FCC4h", "FC4", "FC6", "FT8", "FCC1h", "FCC2h"]
    
    channel_names = ["AFp1", "AFF1h", "FFC1h", "AFp2", "AFF2h", "FFC2h"]

    info = mne.create_info(ch_names=channel_names, sfreq=500, ch_types='eeg')
    montage = mne.channels.make_standard_montage("brainproducts-RNP-BA-128")
    info.set_montage(montage)

    all_avg_band_data = []
    for task, subjects_eeg in nback_tasks.items():
        band_powers = extract_bandpower_data(subjects_eeg)
        avg_band_power = np.mean(band_powers['theta'], axis=0)
        all_avg_band_data.append(avg_band_power)
    
    all_avg_band_data = np.stack(all_avg_band_data)
    vmin, vmax = all_avg_band_data.min(), all_avg_band_data.max()

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.7])  # Colorbar axis

    for i, (task, subjects_eeg) in enumerate(nback_tasks.items()):
        band_powers = extract_bandpower_data(subjects_eeg)
        im = plot_topomaps_avg(band_powers['theta'], task, info, axes[i], vmin, vmax)
    
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Theta Power')

    fig.suptitle(f'Topographic Maps for Average Theta Power - {title_suffix}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.84, 0.95])
    plt.show()

def plot_mean_theta(eeg_results, title_suffix):
    
    subjects_list_theta = []

    for array in eeg_results:
        mean_channels = np.mean(array, axis=-1)
        mean_epochs = np.mean(mean_channels, axis=0)
        subjects_list_theta.append(mean_epochs)
    
    return subjects_list_theta

def plot_mean_hbo(fnirs_results, title_suffix):
    
    subjects_list_hbo = []

    for array in fnirs_results:
        mean_channels = np.mean(array, axis=-1)
        mean_epochs = np.mean(mean_channels, axis=0)
        subjects_list_hbo.append(mean_epochs)
    
    return subjects_list_hbo

def plot_topomap_hbo(raw, fnirs_results, ax, vmin, vmax, cmap='RdBu_r'):
    subjects_list_hbo = []

    for array in fnirs_results:
        mean_epochs = np.mean(array, axis=0)
        subjects_list_hbo.append(mean_epochs)
    
    # Stack the arrays along a new axis (0) to create a 2D array
    stacked_arrays = np.stack(subjects_list_hbo, axis=0)

    # Compute the mean along the new axis (axis=0) to preserve the original shape (4,)
    mean_array = np.mean(stacked_arrays, axis=0)
    print("This is the shape of mean_array:", mean_array.shape)

    ch_idx_by_type = mne.channel_indices_by_type(raw.info)

    # Ensure that the number of data points matches the number of channels
    hbo_info = mne.pick_info(raw.info, sel=ch_idx_by_type["hbo"])
    if mean_array.shape[0] != len(hbo_info['ch_names']):
        raise ValueError(f"The number of HbO channels ({len(hbo_info['ch_names'])}) does not match the data length ({mean_array.shape[0]})")

    im, _ = mne.viz.plot_topomap(mean_array, hbo_info, axes=ax, show=False, vlim=(vmin, vmax), cmap=cmap)

    return im


def plot_violin(data, labels, title, ylabel, width=0.5):
    
    flat_data = [item for sublist in data for item in sublist]
    flat_labels = np.repeat(labels, [len(sublist) for sublist in data])

    plt.figure(figsize=(10,6))
    sns.violinplot(x=flat_labels, y=flat_data, width=width)
    plt.xlabel('N-back Task')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.show()

def plot_boxplot(data, labels, title, ylabel):
    plt.figure(figsize=(10,6))
    plt.boxplot(data, labels = labels)
    plt.xlabel("N-back Task")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

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

        #clean_raw_eeg_hc = "H:\\Dokumenter\\data_acquisition\\data_eeg\\clean_eeg\\healthy_controls\\"
        #clean_raw_eeg_p = "H:\\Dokumenter\\data_acquisition\\data_eeg\\clean_eeg\\patients\\"
        #clean_raw_fnirs_hc = "H:\\Dokumenter\\data_acquisition\\data_fnirs\\healthy_controls\\baseline\\snirf_files\\"
        #clean_raw_fnirs_p = "H:\\Dokumenter\\data_acquisition\\data_fnirs\\patients\\baseline\\snirf_files\\"

        file_dirs_eeg_hc, _ = read_files(clean_raw_eeg_hc, '.fif')
        file_dirs_eeg_p, _ = read_files(clean_raw_eeg_p, '.fif')
        file_dirs_fnirs_hc, _ = read_files(clean_raw_fnirs_hc, '.snirf')
        file_dirs_fnirs_p, _ = read_files(clean_raw_fnirs_p, '.snirf')

        bands = [(0.5, 4, 'Delta'), (4, 8, 'Theta'), (8, 12, 'Alpha'), 
                 (12, 16, 'Sigma'), (16, 30, 'Beta')]
        
        #epoch_duration = 5
        epoch_duration = 8

        # Process EEG data
        eeg_hc_results, subj_hc_bad_epochs = process_eeg_data(file_dirs_eeg_hc, epoch_duration, bands)
        eeg_p_results, subj_p_bad_epochs = process_eeg_data(file_dirs_eeg_p, epoch_duration, bands)

        # Process fNIRS data
        raw_haemo_hc, fnirs_hc_results = process_fnirs_data(file_dirs_fnirs_hc, epoch_duration, subj_hc_bad_epochs)
        raw_haemo_p, fnirs_p_results = process_fnirs_data(file_dirs_fnirs_p, epoch_duration, subj_p_bad_epochs)

        """
        
        One-Sample T-test
        
        """
        
        theta_power_hc = []
        theta_power_p = []

        # Apply one-sample t-test to EEG data
        for i in range(4):
            hc_subjects_theta, t_stat_theta_hc, p_value_theta_hc = one_sample_ttest_eeg(eeg_hc_results["nback"][i])
            p_subjects_theta, t_stat_theta_p, p_value_theta_p = one_sample_ttest_eeg(eeg_p_results["nback"][i])
           
            print(f"{i}-back Healthy Controls t-statistic: {t_stat_theta_hc}, p-value: {p_value_theta_hc}")
            print(f"{i}-back Patients t-statistic: {t_stat_theta_p}, p-value: {p_value_theta_p}")

            theta_power_hc.append(hc_subjects_theta)
            theta_power_p.append(p_subjects_theta)
        
        # Combine data into a single array for theta
        theta_data_hc = np.vstack(theta_power_hc).T
        theta_data_p = np.vstack(theta_power_p).T

        hbo_hc = []
        hbo_p = []
        
        # Apply one-sample t-test to fNIRS data
        for i in range(4):
            hc_subjects_hbo, t_stat_hbo_hc, p_value_hbo_hc = one_sample_ttest_fnirs(fnirs_hc_results["nback"][i])
            p_subjects_hbo, t_stat_hbo_p, p_value_hbo_p = one_sample_ttest_fnirs(fnirs_p_results["nback"][i])
            
            print(f"{i}-back Healthy Controls HBO t-statistic: {t_stat_hbo_hc}, p-value: {p_value_hbo_hc}")
            print(f"{i}-back Patients HBO t-statistic: {t_stat_hbo_p}, p-value: {p_value_hbo_p}")

            hbo_hc.append(hc_subjects_hbo)
            hbo_p.append(p_subjects_hbo)

        # Combine data into a single array for theta
        hbo_data_hc = np.vstack(hbo_hc).T
        hbo_data_p = np.vstack(hbo_p).T

        """
        
        ANOVA
        
        """
        
        import statsmodels.api as sm
        from statsmodels.stats.anova import AnovaRM

        """
        THETA
        """

        # Determine the number of subjects and conditions
        num_subjects_theta_hc, num_conditions_theta_hc = theta_data_hc.shape
        num_subjects_theta_p, num_conditions_theta_p = theta_data_p.shape

        # Create subject IDs based on the number of subjects
        subject_ids_theta_hc = np.arange(1, num_subjects_theta_hc + 1)
        subject_ids_theta_p = np.arange(1, num_subjects_theta_p + 1)

        # Repeat subject IDs for each condition
        repeated_subjects_theta_hc = np.repeat(subject_ids_theta_hc, num_conditions_theta_hc)
        repeated_subjects_theta_p = np.repeat(subject_ids_theta_p, num_conditions_theta_p)

        # Create task conditions array, repeated for each subject
        task_conditions_theta_hc = np.tile(['0-back', '1-back', '2-back', '3-back'], num_subjects_theta_hc)
        task_conditions_theta_p = np.tile(['0-back', '1-back', '2-back', '3-back'], num_subjects_theta_p)

        # Flatten the HBO data array
        theta_values_hc = theta_data_hc.flatten()
        theta_values_p = theta_data_p.flatten()

        # Create a DataFrame
        theta_df_hc = pd.DataFrame({'subject': repeated_subjects_theta_hc, 'task': task_conditions_theta_hc, 'Theta': theta_values_hc})
        theta_df_p = pd.DataFrame({'subject': repeated_subjects_theta_p, 'task': task_conditions_theta_p, 'Theta': theta_values_p})

        # Perform repeated measures ANOVA for HBO values
        theta_rm_anova_hc = AnovaRM(theta_df_hc, 'Theta', 'subject', within=['task'])
        theta_results_hc = theta_rm_anova_hc.fit()
        print("Repeated Measures ANOVA for Theta value HEALTHY CONTROLS:")
        print(theta_results_hc)

        theta_rm_anova_p = AnovaRM(theta_df_p, 'Theta', 'subject', within=['task'])
        theta_results_p = theta_rm_anova_p.fit()
        print("Repeated Measures ANOVA for Theta value PATIENTS:")
        print(theta_results_p)

        """
        HBO
        """
        # Determine the number of subjects and conditions
        num_subjects_hbo_hc, num_conditions_hbo_hc = hbo_data_hc.shape
        num_subjects_hbo_p, num_conditions_hbo_p = hbo_data_p.shape

        # Create subject IDs based on the number of subjects
        subject_ids_hbo_hc = np.arange(1, num_subjects_hbo_hc + 1)
        subject_ids_hbo_p = np.arange(1, num_subjects_hbo_p + 1)

        # Repeat subject IDs for each condition
        repeated_subjects_hbo_hc = np.repeat(subject_ids_hbo_hc, num_conditions_hbo_hc)
        repeated_subjects_hbo_p = np.repeat(subject_ids_hbo_p, num_conditions_hbo_p)

        # Create task conditions array, repeated for each subject
        task_conditions_hbo_hc = np.tile(['0-back', '1-back', '2-back', '3-back'], num_subjects_hbo_hc)
        task_conditions_hbo_p = np.tile(['0-back', '1-back', '2-back', '3-back'], num_subjects_hbo_p)

        # Flatten the HBO data array
        hbo_values_hc = hbo_data_hc.flatten()
        hbo_values_p = hbo_data_p.flatten()

        # Create a DataFrame
        hbo_df_hc = pd.DataFrame({'subject': repeated_subjects_hbo_hc, 'task': task_conditions_hbo_hc, 'HBO': hbo_values_hc})
        hbo_df_p = pd.DataFrame({'subject': repeated_subjects_hbo_p, 'task': task_conditions_hbo_p, 'HBO': hbo_values_p})

        # Perform repeated measures ANOVA for HBO values
        hbo_rm_anova_hc = AnovaRM(hbo_df_hc, 'HBO', 'subject', within=['task'])
        hbo_results_hc = hbo_rm_anova_hc.fit()
        print("Repeated Measures ANOVA for HBO value HEALTHY CONTROLS:")
        print(hbo_results_hc)

        hbo_rm_anova_p = AnovaRM(hbo_df_p, 'HBO', 'subject', within=['task'])
        hbo_results_p = hbo_rm_anova_p.fit()
        print("Repeated Measures ANOVA for HBO value PATIENTS:")
        print(hbo_results_p)

        """
        
        EEG
        
        """
        
        # Plot EEG data for healthy controls
        mean_theta_data_hc = [plot_mean_theta(eeg_hc_results["nback"][i], 'healthy controls') for i in range(4)]
        labels_hc = [f'HC {i}-Back' for i in range(4)]
        plot_violin(mean_theta_data_hc, labels_hc, 'Theta Power for Healthy Controls', 'Theta Power', width=0.3)
        plot_boxplot(mean_theta_data_hc, labels_hc, 'Theta Power for Healthy Controls', 'Theta Power')

        # Plot EEG data for patients
        mean_theta_data_p = [plot_mean_theta(eeg_p_results["nback"][i], 'patients') for i in range(4)]
        labels_p = [f'P {i}-Back' for i in range(4)]
        plot_violin(mean_theta_data_p, labels_p, 'Theta Power for Patients', 'Theta Power', width=0.3)
        plot_boxplot(mean_theta_data_p, labels_p, 'Theta Power for Patients', 'Theta Power')

        """
        
        fNIRS
        
        """
        
        # Define conditions and plot topomaps for each condition and group
        conditions = ['0-back', '1-back', '2-back', '3-back']

        # Gather all data to compute global vmin and vmax
        all_mean_data_hbo = []

        for i, condition in enumerate(conditions):
            for group_data in [fnirs_hc_results["nback"][i], fnirs_p_results["nback"][i]]:
                subjects_list_hbo = []
                for array in group_data:
                    mean_epochs = np.mean(array, axis=0)
                    subjects_list_hbo.append(mean_epochs)
                
                stacked_arrays = np.stack(subjects_list_hbo, axis=0)
                mean_array = np.mean(stacked_arrays, axis=0)
                #mean_data_hbo = np.mean(mean_array, axis=0)
                all_mean_data_hbo.append(mean_array)

        all_mean_data_hbo = np.concatenate(all_mean_data_hbo, axis=0)
        global_min_hbo = np.min(all_mean_data_hbo)
        global_max_hbo = np.max(all_mean_data_hbo)

        vmin = global_min_hbo
        vmax = global_max_hbo
        cmap = 'RdBu_r'

        # Adding topomap for HbO and HbR
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle('HbO Topomaps', fontsize=20)

        for i, condition in enumerate(conditions):
            # Plot for Healthy Controls
            ax = axes[0, i]
            im = plot_topomap_hbo(raw_haemo_hc, fnirs_hc_results["nback"][i], ax, vmin, vmax, cmap)
            ax.set_title(f'HbO {condition} - Healthy Controls')

            # Plot for Patients
            ax = axes[1, i]
            im = plot_topomap_hbo(raw_haemo_p, fnirs_p_results["nback"][i], ax, vmin, vmax, cmap)
            ax.set_title(f'HbO {condition} - Patients')

        # Add a single color bar for all topomaps
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax)

        # Create custom legend for topomaps
        topomap_legend_patches = [Patch(color='white', label='HbO Mean Value')]
        fig.legend(handles=topomap_legend_patches, loc='upper left')
        plt.tight_layout(rect=[0, 0, 0.9, 0.95])
        plt.show()
        
        # Plot fNIRS data for healthy controls
        mean_hbo_data_hc = [plot_mean_hbo(fnirs_hc_results["nback"][i], 'healthy controls') for i in range(4)]
        plot_violin(mean_hbo_data_hc, labels_hc, 'HBO Concentration for Healthy Controls', 'HBO Concentration', width=0.3)
        plot_boxplot(mean_hbo_data_hc, labels_hc, 'HBO Concentration for Healthy Controls', 'HBO Concentration')

        # Plot fNIRS data for patients
        mean_hbo_data_p = [plot_mean_hbo(fnirs_p_results["nback"][i], 'patients') for i in range(4)]
        plot_violin(mean_hbo_data_p, labels_p, 'HBO Concentration for Patients', 'HBO Concentration', width=0.3)
        plot_boxplot(mean_hbo_data_p, labels_p, 'HBO Concentration for Patients', 'HBO Concentration')

        eeg_features(eeg_hc_results["nback"][0], eeg_hc_results["nback"][1], eeg_hc_results["nback"][2], eeg_hc_results["nback"][3], title_suffix='healthy controls')
        eeg_features(eeg_p_results["nback"][0], eeg_p_results["nback"][1], eeg_p_results["nback"][2], eeg_p_results["nback"][3], title_suffix='patients')

    main()