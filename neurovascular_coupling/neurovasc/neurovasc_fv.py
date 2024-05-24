import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import mne
from scipy.integrate import simps
from scipy.signal import welch

from arrange_files import read_files
from Hemo import HemoData
from FeatureExtraction import FeatureExtraction

# Function to empty a directory
def empty_directory(directory_path):
    if os.path.exists(directory_path):
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
    else:
        os.makedirs(directory_path)
        print(f'Directory {directory_path} created.')

"""

Characterize data

"""

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

    for i in delay:
        #print("This is the number", i)
        epochs = list()
        events_0back = mne.make_fixed_length_events(raw_0back, start = 0 , stop = 58 + epoch_duration - i, duration = epoch_duration)
        events_1back = mne.make_fixed_length_events(raw_1back, start = 0 , stop = 70 + epoch_duration - i, duration = epoch_duration)
        events_2back = mne.make_fixed_length_events(raw_2back, start = 0 , stop = 70 + epoch_duration - i, duration = epoch_duration)
        events_3back = mne.make_fixed_length_events(raw_3back, start = 0 , stop = 60 + epoch_duration - i, duration = epoch_duration)
        
        interval = (0,0)
        epochs_0back = mne.Epochs(raw_0back, events_0back, baseline = interval, preload = True)
        epochs_1back = mne.Epochs(raw_1back, events_1back, baseline = interval, preload = True)
        epochs_2back = mne.Epochs(raw_2back, events_2back, baseline = interval, preload = True)
        epochs_3back = mne.Epochs(raw_3back, events_3back, baseline = interval, preload = True)

        epochs.append(epochs_0back)
        epochs.append(epochs_1back)
        epochs.append(epochs_2back)
        epochs.append(epochs_3back)

        eeg_epochs_coupling.append(epochs)
    
    #print("This is the length eeg_epochs", len(eeg_epochs_coupling))
    #print("This is the length eeg_epochs", eeg_epochs_coupling)

    return eeg_epochs_coupling

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

# EEG

def process_eeg(file_dirs_eeg, bands, epoch_duration):
    subjects = []
    for file in file_dirs_eeg:
        raw = mne.io.read_raw_fif(file)
        raw.load_data()
        epochs = characterization_eeg(raw, epoch_duration)
        
        bp_relative = []
        for coupling in epochs:
            channel_drop = [ 'AF7', 'AFF5h', 'FT7', 'FC5', 'FC3', 'FCC3h', 
                    'CCP3h', 'CCP1h', 'CP1', 'TP7', 'CPP3h', 'P1', 
                    'AF8', 'AFF6h', 'FT8', 'FC6', 'FC4', 'FCC4h', 'CCP4h', 
                    'CCP2h', 'CP2', 'P2', 'CPP4h', 'TP8']
        
            epochs = [c.drop_channels(channel_drop) for c in coupling]

            concatenated_data = [c.get_data(units="uV") for c in epochs]
            data_eeg = np.concatenate(concatenated_data, axis=0)
            sf = coupling[0].info['sfreq']
            freqs, psd = welch(data_eeg, sf, nperseg=int(4 * sf))
            bp_relative.append(bandpower_from_psd_ndarray(psd, freqs, bands, ln_normalization=False, relative=True))

        coupling_data = [np.expand_dims(bp, axis=0) for bp in bp_relative]
        subjects.append(coupling_data)

    delays = [np.concatenate([s[j] for s in subjects], axis=0) for j in range(10)]

    #for delay in delays:
    #    print("This is the shape of the delay", delay.shape)

    return delays

def calculate_mean_eeg(delays):
    mean_values = []
    for delay in delays:
        theta_bp_relative = delay[:, 1, :, :] #(subj, freq band, epochs, channels)
        mean_channels = np.mean(np.mean(theta_bp_relative, axis=0), axis=-1)
        mean_epochs = np.mean(mean_channels, axis=0)
        mean_values.append(mean_epochs)
    
    return mean_values

# fNIRS

def process_fnirs_data(file_dirs_fnirs, epoch_duration):
    subjects = []
    for file_dir in file_dirs_fnirs:
        raw_haemo = HemoData(file_dir, preprocessing=True, isPloting=False).getMneIoRaw()
        epochs_fnirs = characterization_fNIRS(raw_haemo, epoch_duration)

        features_fnirs = []
        for coupling in epochs_fnirs:
            concatenated_data = np.concatenate([c.get_data(picks=["hbo"]) for c in coupling], axis=0)
            mean_std_per_epoch_per_channel = np.concatenate([
                np.expand_dims(np.mean(concatenated_data, axis=-1), axis=-1),
                np.expand_dims(np.std(concatenated_data, axis=-1), axis=-1)], axis=-1)
            features_fnirs.append(np.expand_dims(mean_std_per_epoch_per_channel, axis=0))
        subjects.append(features_fnirs)

    delays = [np.concatenate([subject[j] for subject in subjects], axis=0) for j in range(10)]

    mean_values = []
    for delay in delays:
        #print(f"This is the shape of delay: {delay.shape}")
        hbo_mean = delay[:, :, :, 0]
        mean_across_subjects = np.mean(hbo_mean, axis=0)
        mean_across_channels = np.mean(mean_across_subjects, axis=-1)
        mean_across_epochs = np.mean(mean_across_channels, axis=0)
        mean_values.append(mean_across_epochs)
        #print(f"This is the shape of mean_values: {mean_across_epochs.shape}")

    return mean_values

# Set up directories
os.chdir("/Users/adriana/Documents/GitHub/thesis")
mne.set_log_level('error')
mne.set_config('MNE_BROWSER_BACKEND', 'matplotlib')

# EEG
clean_raw_eeg_hc = "/Users/adriana/Documents/DTU/thesis/data_acquisition/clean_eeg/healthy_controls/"
clean_raw_eeg_p = "/Users/adriana/Documents/DTU/thesis/data_acquisition/clean_eeg/patients/"
# fNIRS
clean_raw_fnirs_hc = "/Users/adriana/Documents/DTU/thesis/data_acquisition/data_fnirs/healthy_controls/"
clean_raw_fnirs_p = "/Users/adriana/Documents/DTU/thesis/data_acquisition/data_fnirs/patients/"

# EEG
file_dirs_eeg_hc, hc_eeg = read_files(clean_raw_eeg_hc, '.fif')
file_dirs_eeg_p, p_eeg = read_files(clean_raw_eeg_p, '.fif')
# fNIRS
file_dirs_fnirs_hc, _ = read_files(clean_raw_fnirs_hc, '.snirf')
file_dirs_fnirs_p, _ = read_files(clean_raw_fnirs_p, '.snirf')

bands = [(0.5, 4, 'Delta'), (4, 8, 'Theta'), (8, 12, 'Alpha'), 
         (12, 16, 'Sigma'), (16, 30, 'Beta')]

epoch_duration = 1
delay_time = np.arange(0, 10, epoch_duration)

"""
EEG
"""
eeg_delays_hc = process_eeg(file_dirs_eeg_hc, bands, epoch_duration)
eeg_delays_p = process_eeg(file_dirs_eeg_p, bands, epoch_duration)

eeg_hc_mean = calculate_mean_eeg(eeg_delays_hc)
eeg_p_mean = calculate_mean_eeg(eeg_delays_p)

# Plot results
plt.plot(delay_time, eeg_p_mean, label="patients")
plt.plot(delay_time, eeg_hc_mean, label="healthy")
plt.xlabel("delay")
plt.ylabel("relative theta power")
plt.legend()
plt.show()

"""
fNIRS
"""

fnirs_hc_mean = process_fnirs_data(file_dirs_fnirs_hc, epoch_duration)
fnirs_p_mean = process_fnirs_data(file_dirs_fnirs_p, epoch_duration)

plt.plot(delay_time, fnirs_p_mean, label="patients")
plt.plot(delay_time, fnirs_hc_mean, label="healthy")
plt.xlabel("delay")
plt.ylabel("hbo concentration")
plt.legend()
plt.show()

"""

COUPLING

"""

import scipy
from scipy.stats import pearsonr, spearmanr


"""
Healthy controls
"""
hc_correlation_list = list()
hc_p_value_list = list()

# Calculate correlation
for fnirs, eeg in zip(fnirs_hc_mean, eeg_hc_mean):
    correlation_hc, p_value_hc = pearsonr(fnirs, eeg)
    hc_correlation_list.append(correlation_hc)
    hc_p_value_list.append(p_value_hc)

plt.plot(hc_correlation_list, delay_time)
plt.xlabel("delay")
plt.ylabel("correlation coefficient")
plt.title("Healthy Controls")
plt.show()

"""
Patients
"""

p_correlation_list = list()
p_p_value_list = list()

# Calculate correlation
for fnirs, eeg in zip(fnirs_p_mean, eeg_p_mean):
    correlation_p, p_value_p = pearsonr(fnirs, eeg)
    p_correlation_list.append(correlation_p)
    p_p_value_list.append(p_value_p)

plt.plot(p_correlation_list, delay_time)
plt.xlabel("delay")
plt.ylabel("correlation coefficient")
plt.title("Patients")
plt.show()


