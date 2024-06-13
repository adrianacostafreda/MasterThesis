import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import mne
from scipy.integrate import simps
from scipy.signal import welch
from scipy.signal import spectrogram, resample
from autoreject import AutoReject, get_rejection_threshold

import seaborn as sns
sns.set(style='white', font_scale=1.2)

#from basic.arrange_files import read_files
#from neurovascular_coupling.Hemo import HemoData
from arrange_files import read_files
from Hemo import HemoData

"""

Characterize data

"""

# Function to visualize dropped epochs
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

    raw = raw_new.copy().crop(tmin=trigger_data['4']['begin'][0], tmax=trigger_data['7']['end'][0] + 1)

    # Epoch data

    """
    Returns a mne.Epochs object created from the annotated mne.io.Raw object.

    """

    delay = np.arange(0 + epoch_duration, 10 + epoch_duration, epoch_duration)

    eeg_epochs_coupling = list()
    eeg_bad_epochs = list()

    for i in delay:
        eeg_epochs = []

        events = mne.make_fixed_length_events(raw, start = 0 + i, stop = raw.times[-1] , duration = epoch_duration)
        
        
        interval = (0,0)
        epochs = mne.Epochs(raw, events, baseline = interval, preload = True)

        # -----------------Get-Rejection-Threshold---------------------------------
        #reject_thresholds = get_rejection_threshold(epochs, ch_types = "eeg", verbose = False)
            
        #epochs.drop_bad(reject=reject_thresholds)

        # Visualize dropped epochs
        #bad_epoch = visualize_dropped_epochs(epochs)
            
        eeg_epochs.append(epochs)
        #eeg_bad_epochs.append(bad_epoch)
        eeg_epochs_coupling.append(eeg_epochs)

    return (eeg_epochs_coupling, eeg_bad_epochs)

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

    raw = raw_new.copy().crop(tmin=trigger_data['4']['begin'][0], tmax=trigger_data['7']['end'][0] + 1)

    # Epoch data

    #delay = np.arange(epoch_duration, 10 + epoch_duration, epoch_duration)
    delay = np.arange(0 + epoch_duration, 10 + epoch_duration, epoch_duration)

    fnirs_epochs_coupling = list()

    # make a 10 s delay
    #for i, bad in zip(delay, bad_epochs):
    for i in delay:
        epochs_fnirs = list()
        events = mne.make_fixed_length_events(raw, start = 0, stop= raw.times[-1] + epoch_duration -i, duration = epoch_duration)
        
        interval = (0,0)
        epochs = mne.Epochs(raw, events, baseline = interval, preload = True)
        #epochs.drop(bad[0])

        epochs_fnirs.append(epochs)
        fnirs_epochs_coupling.append(epochs_fnirs)

    return fnirs_epochs_coupling

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

def process_fnirs_data(file_dirs_fnirs, epoch_duration, samples, bad_epochs):
    
    subjects = []
    for file_dir, bad_epoch in zip(file_dirs_fnirs, bad_epochs):
        
        raw_haemo = HemoData(file_dir, preprocessing=True, isPloting=False).getMneIoRaw()
        
        epochs_fnirs = characterization_fNIRS(raw_haemo, epoch_duration, bad_epoch)

        features_fnirs = []
        for coupling in epochs_fnirs:

            fnirs_data = [c.get_data(picks=["hbo"]) for c in coupling]
            
            mean_std_per_epoch_per_channel = np.concatenate([
                np.expand_dims(np.mean(fnirs_data, axis=-1), axis=-1),
                np.expand_dims(np.std(fnirs_data, axis=-1), axis=-1)], axis=-1)
            features_fnirs.append(mean_std_per_epoch_per_channel)
        
        subjects.append(features_fnirs)

    return subjects

# Set up directories
os.chdir("/Users/adriana/Documents/GitHub/thesis")

#os.chdir("H:\Dokumenter\GitHub\MasterThesis\.venv")
mne.set_log_level('error')
mne.set_config('MNE_BROWSER_BACKEND', 'matplotlib')

# EEG
clean_raw_eeg_hc = "/Users/adriana/Documents/DTU/thesis/data_acquisition/clean_eeg/healthy_controls/"
clean_raw_eeg_p = "/Users/adriana/Documents/DTU/thesis/data_acquisition/clean_eeg/patients/"

#clean_raw_eeg_hc = "H:\\Dokumenter\\data_acquisition\\data_eeg\\clean_eeg\\healthy_controls\\block1\\"
#clean_raw_eeg_p = "H:\\Dokumenter\\data_acquisition\\data_eeg\\clean_eeg\\patients\\block1\\"

# fNIRS
clean_raw_fnirs_hc = "/Users/adriana/Documents/DTU/thesis/data_acquisition/data_fnirs/healthy_controls/"
clean_raw_fnirs_p = "/Users/adriana/Documents/DTU/thesis/data_acquisition/data_fnirs/patients/"

#clean_raw_fnirs_hc = "H:\\Dokumenter\\data_acquisition\\data_fnirs\\healthy_controls\\baseline\\snirf_files\\block1\\"
#clean_raw_fnirs_p = "H:\\Dokumenter\\data_acquisition\\data_fnirs\\patients\\baseline\\snirf_files\\block1\\"

# EEG
file_dirs_eeg_hc, hc_eeg = read_files(clean_raw_eeg_hc, '.fif')
file_dirs_eeg_p, p_eeg = read_files(clean_raw_eeg_p, '.fif')

# fNIRS
file_dirs_fnirs_hc, _ = read_files(clean_raw_fnirs_hc, '.snirf')
file_dirs_fnirs_p, _ = read_files(clean_raw_fnirs_p, '.snirf')

bands = [(0.5, 4, 'Delta'), (4, 8, 'Theta'), (8, 12, 'Alpha'), 
         (12, 16, 'Sigma'), (16, 30, 'Beta')]

epoch_duration = 5
end_time = 10

delay_time = np.arange(0, end_time, epoch_duration)

delay_eeg = np.arange(0 + epoch_duration, end_time + epoch_duration, epoch_duration)
delay_fnirs = np.arange(0 + epoch_duration, end_time + epoch_duration, epoch_duration)

samples_eeg = len(delay_eeg)
samples_fnirs = len(delay_fnirs)

"""
EEG
"""

# HEALTHY CONTROLS
subjects_eeg_hc = []
subjects_eeg_hc_bad_epochs = []

for file_hc in file_dirs_eeg_hc:
    raw = mne.io.read_raw_fif(file_hc)
    raw.load_data()

    characterize_eeg_hc = characterization_eeg(raw, epoch_duration)

    epochs_hc = characterize_eeg_hc[0]
    bad_epochs_hc = characterize_eeg_hc[1]
        
    bp_relative_hc = []

    for coupling_hc in epochs_hc:
        channel_drop = [ 'AF7', 'AFF5h', 'FT7', 'FC5', 'FC3', 'FCC3h', 
                    'CCP3h',  'CP1', 'TP7', 'CPP3h', 'P1', 
                    'AF8', 'AFF6h', 'FT8', 'FC6', 'FC4', 'FCC4h', 'CCP4h', 
                     'CP2', 'P2', 'CPP4h', 'TP8']
        
        epochs_hc = [c.drop_channels(channel_drop) for c in coupling_hc]

        concatenated_data_hc = [c.get_data(units="uV") for c in epochs_hc]
        data_eeg_hc = np.concatenate(concatenated_data_hc, axis=0)
        
        sf = coupling_hc[0].info['sfreq']
        nperseg=256
        noverlap = 0.5*256
        freqs_hc, psd_hc = welch(data_eeg_hc, sf, nperseg=256, noverlap=0.5*256)
        psd_hc_mean_epoch = np.mean(psd_hc, axis=0)
        psd_hc_mean_chan = np.mean(psd_hc_mean_epoch, axis=0)

        # Plot
        plt.plot(freqs_hc, psd_hc_mean_chan , 'k', lw=2)
        #plt.fill_between(freqs_hc, psd_hc_mean_chan , cmap='Spectral')
        plt.xlim(0, 30)
        #plt.yscale('log')
        sns.despine()
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('PSD log($uV^2$/Hz)')
        #plt.show()
        
        bp_relative_hc.append(bandpower_from_psd_ndarray(psd_hc, freqs_hc, bands, ln_normalization=False, relative=True))

    subjects_eeg_hc.append(bp_relative_hc)
    subjects_eeg_hc_bad_epochs.append(bad_epochs_hc)

for subject_eeg_hc in subjects_eeg_hc:
    for delay_eeg_hc in subject_eeg_hc:
        print("This is the shape of the hc delay EEG", delay_eeg_hc.shape)

# PATIENTS
subjects_eeg_p = []
subjects_eeg_p_bad_epochs = []

for file_p in file_dirs_eeg_p:
    raw = mne.io.read_raw_fif(file_p)
    raw.load_data()
    
    characterize_eeg_p = characterization_eeg(raw, epoch_duration)

    epochs_p = characterize_eeg_p[0]
    bad_epochs_p = characterize_eeg_p[1]
        
    bp_relative_p = []
    for coupling_p in epochs_p:
        channel_drop = ['AF7', 'AFF5h', 'FT7', 'FC5', 'FC3', 'FCC3h', 
                    'CCP3h', 'CCP1h', 'CP1', 'TP7', 'CPP3h', 'P1', 
                    'AF8', 'AFF6h', 'FT8', 'FC6', 'FC4', 'FCC4h', 'CCP4h', 
                    'CCP2h', 'CP2', 'P2', 'CPP4h', 'TP8']
        
        epochs_p = [c.drop_channels(channel_drop) for c in coupling_p]

        concatenated_data_p = [c.get_data(units="uV") for c in epochs_p]
        data_eeg_p = np.concatenate(concatenated_data_p, axis=0)

        sf = coupling_p[0].info['sfreq']
        nperseg=256
        noverlap = 0.5*256
        freqs_p, psd_p = welch(data_eeg_p, sf, nperseg=256, noverlap=0.5*256)

        psd_p_mean_epoch = np.mean(psd_p, axis=0)
        psd_p_mean_chan = np.mean(psd_p_mean_epoch, axis=0)

        # Plot
        plt.plot(freqs_p, psd_p_mean_chan, 'k', lw=2)
        #plt.fill_between(freqs_p, psd_p_mean_chan, cmap='Spectral')
        plt.xlim(0, 50)
        #plt.yscale('log')
        sns.despine()
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('PSD log($uV^2$/Hz)')
        #plt.show()
        
        bp_relative_p.append(bandpower_from_psd_ndarray(psd_p, freqs_p, bands, ln_normalization=False, relative=True))

    subjects_eeg_p.append(bp_relative_p)
    subjects_eeg_p_bad_epochs.append(bad_epochs_p)

for subject_eeg_p in subjects_eeg_p:
    for delay_eeg_p in subject_eeg_p:
        print("This is the shape of the patient delay EEG", delay_eeg_p.shape)

subjects_fnirs_hc = process_fnirs_data(file_dirs_fnirs_hc, epoch_duration, samples_fnirs, subjects_eeg_hc_bad_epochs)
subjects_fnirs_p = process_fnirs_data(file_dirs_fnirs_p, epoch_duration, samples_fnirs, subjects_eeg_p_bad_epochs)

for subject_fnirs_hc in subjects_fnirs_hc:
    for delay_fnirs_hc in subject_fnirs_hc:
        print("This is the shape of the delay hc fnirs", delay_fnirs_hc.shape)

for subject_fnirs_p in subjects_fnirs_p:
    for delay_fnirs_p in subject_fnirs_p:
        print("This is the shape of the delay patients fnirs", delay_fnirs_p.shape)

from scipy.stats import pearsonr

def compute_and_plot_correlations(subjects_eeg, subjects_fnirs, delay_time, title_suffix):
    
    correlation_eeg_subj = []
    correlation_fnirs_subj = []

    for subject_eeg in subjects_eeg:
        correlation_eeg = []
        for delay_eeg in subject_eeg:
            theta = delay_eeg[1, :, :]
            mean_chann = np.mean(theta, axis=-1)
            correlation_eeg.append(mean_chann)
        print("This is the length of correlation eeg", len(correlation_eeg))
        correlation_eeg_subj.append(correlation_eeg)

    for subject_fnirs in subjects_fnirs:
        correlation_fnirs = []
        for delay_fnirs in subject_fnirs:
            
            mean_hbo = delay_fnirs[:,:, 0:4, 0]
            mean_chann_hbo = np.mean(mean_hbo, axis=-1)
            mean_subj_hbo = np.mean(mean_chann_hbo, axis=0)
            correlation_fnirs.append(mean_subj_hbo)
        correlation_fnirs_subj.append(correlation_fnirs)

    print("This is the length of subj correlation_eeg", len(correlation_eeg_subj))
    print("This is the length of subj correlation_fnirs", len(correlation_fnirs_subj))

    correlation_results = []
    p_values = []
    
    for eeg_subj, fnirs_subj in zip(correlation_eeg_subj, correlation_fnirs_subj):
        subject_correlations = []
        subject_p_values = []
        
        for eeg_array, fnirs_array in zip(eeg_subj, fnirs_subj):
            print("This is the shape of eeg_array and fnirs_array", eeg_array.shape, fnirs_array.shape)
            correlation_coefficient, p_value = pearsonr(eeg_array, fnirs_array)
            subject_correlations.append(correlation_coefficient)
            subject_p_values.append(p_value)
        
        correlation_results.append(subject_correlations)
        p_values.append(subject_p_values)
        print("pvalues", p_values)

    plt.figure(figsize=(14, 6))

    # Number of elements in each list (assuming all lists are of equal length)
    num_elements = len(correlation_results[0])

    # Compute the mean of each element position across the lists
    mean_values = []
    for i in range(num_elements):
        element_sum = sum(subject_idx[i] for subject_idx in correlation_results)
        mean_value = element_sum / len(correlation_results)
        mean_values.append(mean_value)

    # Compute the standard deviation of each element position across the lists
    std_devs = []
    for i in range(num_elements):
        variance = sum((subject_idx[i] - mean_values[i]) ** 2 for subject_idx in correlation_results) / len(correlation_results)
        std_dev = np.sqrt(variance)
        std_devs.append(std_dev)
    
    # Convert lists to numpy arrays for easier manipulation
    mean_values = np.array(mean_values)
    std_devs = np.array(std_devs)

    # Define the positions (assuming delay_time is provided or generated)
    delay_time = np.arange(num_elements)

    # Plot the mean values with shaded area representing the standard deviation
    plt.figure(figsize=(10, 6))
    plt.plot(delay_time, mean_values, '-o', label='Mean ' + title_suffix)
    plt.fill_between(delay_time, mean_values - std_devs, mean_values + std_devs, color='b', alpha=0.2, label='Std Dev ' + title_suffix)
    plt.xlabel("Time Delay")
    plt.ylabel("Pearson Correlation Coefficient")
    plt.title("Pearson Correlation Coefficients vs. Time Delays " + title_suffix)
    plt.legend()
    plt.tight_layout()
    plt.show()

def feature_correlations(subjects_eeg, subjects_fnirs, delay_time, title_suffix):
    
    correlation_eeg_subj = []
    correlation_fnirs_subj = []

    for subject_eeg in subjects_eeg:
        correlation_eeg = []
        for delay_eeg in subject_eeg:
            theta = delay_eeg[1, :, :]
            mean_chann = np.mean(theta, axis=-1)
            correlation_eeg.append(mean_chann)
        
        correlation_eeg_subj.append(correlation_eeg)

    for subject_fnirs in subjects_fnirs:
        correlation_fnirs = []
        for delay_fnirs in subject_fnirs:
            mean_hbo = delay_fnirs[:,:, 0:4, 0]
            mean_chann_hbo = np.mean(mean_hbo, axis=-1)
            mean_subj_hbo = np.mean(mean_chann_hbo, axis=0)
            correlation_fnirs.append(mean_subj_hbo)
        correlation_fnirs_subj.append(correlation_fnirs)

    hbo_results = []
    theta_results = []
    
    for eeg_subj, fnirs_subj in zip(correlation_eeg_subj, correlation_fnirs_subj):
        subject_hbo = []
        subject_theta = []
        
        for eeg_array, fnirs_array in zip(eeg_subj, fnirs_subj):
            theta_mean = np.mean(eeg_array, axis=0)
            subject_theta.append(theta_mean)
            hbo_mean = np.mean(fnirs_array, axis=0)
            subject_hbo.append(hbo_mean)
        
        hbo_results.append(subject_hbo)
        theta_results.append(subject_theta)

    plt.figure(figsize=(14, 6))

        # Number of elements in each list (assuming all lists are of equal length)
    num_elements = len(theta_results[0])

    # Compute the mean of each element position across the lists
    mean_values_theta = []
    for i in range(num_elements):
        element_sum = sum(subject_idx[i] for subject_idx in theta_results)
        mean_value = element_sum / len(theta_results)
        mean_values_theta.append(mean_value)
    
    # Compute the standard deviation of each element position across the lists
    std_devs_theta = []
    for i in range(num_elements):
        variance_theta = sum((subject_idx[i] - mean_values_theta[i]) ** 2 for subject_idx in theta_results) / len(theta_results)
        std_dev_theta = np.sqrt(variance_theta)
        std_devs_theta.append(std_dev_theta)

    # Convert to numpy arrays for easier manipulation
    mean_values_theta = np.array(mean_values_theta)
    std_devs_hbo = np.array(std_devs_theta)

    # Define the positions
    positions = np.arange(num_elements)

    # Define the positions
    positions = np.arange(num_elements)

    # Plot the mean values with shaded area representing the standard deviation
    plt.figure(figsize=(10, 6))
    plt.plot(positions, mean_values_theta, '-o', label='Mean ' + title_suffix)
    #plt.fill_between(positions, mean_values_theta - std_devs_theta, mean_values_theta + std_devs_theta, color='b', alpha=0.2, label='Std Dev')
    plt.xlabel('Delay (s)')
    plt.ylabel('Theta Power')
    plt.title('Mean and Standard Deviation Theta Power ' + title_suffix)
    plt.legend()
    plt.grid(True)
    plt.show()

    # Compute the mean of each element position across the lists
    mean_values_hbo = []
    for i in range(num_elements):
        element_sum = sum(subject_idx[i] for subject_idx in hbo_results)
        mean_value = element_sum / len(hbo_results)
        mean_values_hbo.append(mean_value)

    # Compute the standard deviation of each element position across the lists
    std_devs_hbo = []
    for i in range(num_elements):
        variance_hbo = sum((subject_idx[i] - mean_values_hbo[i]) ** 2 for subject_idx in hbo_results) / len(hbo_results)
        std_dev_hbo = np.sqrt(variance_hbo)
        std_devs_hbo.append(std_dev_hbo)
    
    # Convert to numpy arrays for easier manipulation
    mean_values_hbo = np.array(mean_values_hbo)
    std_devs_hbo = np.array(std_devs_hbo)

    # Define the positions
    positions = np.arange(num_elements)

    # Plot the mean values with shaded area representing the standard deviation
    plt.figure(figsize=(10, 6))
    plt.plot(positions, mean_values_hbo, '-o', label='Mean ' + title_suffix)
    plt.fill_between(positions, mean_values_hbo - std_devs_hbo, mean_values_hbo + std_devs_hbo, color='b', alpha=0.2, label='Std Dev '+ title_suffix)
    plt.xlabel('Delay (s)')
    plt.ylabel('HBR Concentration')
    plt.title('Mean and Standard Deviation HBR Concentration ' + title_suffix)
    plt.legend()
    plt.grid(True)
    plt.show()

# Compute and plot correlations for healthy controls
compute_and_plot_correlations(subjects_eeg_hc, subjects_fnirs_hc, delay_time, "Healthy Controls")
compute_and_plot_correlations(subjects_eeg_p, subjects_fnirs_p, delay_time, "Patients")

feature_correlations(subjects_eeg_hc, subjects_fnirs_hc, delay_time, "Healthy Controls")
feature_correlations(subjects_eeg_p, subjects_fnirs_p, delay_time, "Patients")
