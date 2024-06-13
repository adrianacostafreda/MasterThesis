import mne
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
from scipy.integrate import simps
from scipy.signal import welch
import yasa
from mnelab.io.xdf import read_raw_xdf

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

file_dirs_eeg_hc, subjects_hc_eeg = read_files(eeg_dir_inprogress_hc,'.fif')
file_dirs_eeg_p, subjects_p_eeg = read_files(eeg_dir_inprogress_p,'.fif')

bands = [(0.5, 4, 'Delta'), (4, 8, 'Theta'), (8, 12, 'Alpha'), 
        (12, 16, 'Sigma'), (16, 30, 'Beta')]

mne.set_config('MNE_BROWSER_BACKEND', 'matplotlib')

delay_time = np.arange(0, 10, epoch_duration)

# Healthy controls
subjects_hc = list()

# Loop through all the subjects' directories (EEG files directories)
for i in range(len(file_dirs_eeg_hc)):
    
    # --------------Read Raw data -------------------
    raw_hc = mne.io.read_raw_fif(file_dirs_eeg_hc[i])
    raw_hc.load_data()

    # --------------Epoched Data--------------------------------
    # Each epoched object contains 1s epoched data corresponding to each data segment

    epochs_eeg_hc = characterization_eeg(raw_hc, epoch_duration)
    #print("This is the list containing the epochs", epochs_eeg_hc)

    hc_bp_relative = list()

    # --------------Extract features--------------------------------
    for coupling in epochs_eeg_hc:
        concatenated_data = list()
        for n_back in range(len(coupling)):
            #print("This is each segment of the epoched data with delay", coupling[i])
            
            channel_drop = [ 'AF7', 'AFF5h', 'FT7', 'FC5', 'FC3', 'FCC3h', 
                    'CCP3h', 'CCP1h', 'CP1', 'TP7', 'CPP3h', 'P1', 
                    'AF8', 'AFF6h', 'FT8', 'FC6', 'FC4', 'FCC4h', 'CCP4h', 
                    'CCP2h', 'CP2', 'P2', 'CPP4h', 'TP8']

            coupling[n_back].drop_channels(channel_drop)

            # Create a 3-D array
            data = coupling[n_back].get_data(units="uV")
            sf = coupling[n_back].info['sfreq']

            win = int(4 * sf)  # Window size is set to 4 seconds

            concatenated_data.append(data)
            
        data_eeg = np.concatenate(concatenated_data, axis = 0) # shape is (epochs, channels, samples)
        print("This is the shape of data_eeg", data_eeg.shape)

        # Compute the relative power
        freqs, psd = welch(data_eeg, sf, nperseg=win) 
        bp_relative = bandpower_from_psd_ndarray(psd, freqs, bands, ln_normalization = False, relative = True)

        hc_bp_relative.append(bp_relative)
    
    coupling_hc = list()
    for coupling in hc_bp_relative:
        
        coupling = np.expand_dims(coupling, axis=0)
        #print("This is the shape of coupling", coupling.shape)
        coupling_hc.append(coupling)

    subjects_hc.append(coupling_hc)

delays_hc = [list() for _ in range(10)]

for i in subjects_hc:
    for j in range(10):
        delays_hc[j].append(i[j])

delays_hc = [np.concatenate(delay, axis=0) for delay in delays_hc]

hc_mean = list()
for delay in delays_hc:
    #print("This is the shape", delay.shape)
    theta_bp_relative_hc = delay[:, 1, : , :] #(subj, freq band, epochs, channels)
    mean_hc = np.mean(theta_bp_relative_hc, axis=0)
    #print("This is the shape of mean_hc", mean_hc.shape)
    mean_channels_hc = np.mean(mean_hc, axis=-1)
    #print("This is the shape of mean_channels_hc", mean_channels_hc.shape)
    mean_epochs_hc = np.mean(mean_channels_hc, axis=0)
    print("This is mean_epochs_hc", mean_epochs_hc)
    hc_mean.append(mean_epochs_hc)

print("This is the length of hc_mean", hc_mean)


# Patients
subjects_p = list()

# Loop through all the subjects' directories (EEG files directories)
for i in range(len(file_dirs_eeg_p)):
    
    # --------------Read Raw data -------------------
    raw_p = mne.io.read_raw_fif(file_dirs_eeg_p[i])
    raw_p.load_data()

    # --------------Epoched Data--------------------------------
    # Each epoched object contains 1s epoched data corresponding to each data segment

    epochs_eeg_p = characterization_eeg(raw_p, epoch_duration)
    #print("This is the list containing the epochs", epochs_eeg_p)

    p_bp_relative = list()

    # --------------Extract features--------------------------------
    for coupling in epochs_eeg_p:
        concatenated_data = list()
        for n_back in range(len(coupling)):
            #print("This is each segment of the epoched data with delay", coupling[i])
            
            channel_drop = [ 'AF7', 'AFF5h', 'FT7', 'FC5', 'FC3', 'FCC3h', 
                    'CCP3h', 'CCP1h', 'CP1', 'TP7', 'CPP3h', 'P1', 
                    'AF8', 'AFF6h', 'FT8', 'FC6', 'FC4', 'FCC4h', 'CCP4h', 
                    'CCP2h', 'CP2', 'P2', 'CPP4h', 'TP8']

            coupling[n_back].drop_channels(channel_drop)

            # Create a 3-D array
            data = coupling[n_back].get_data(units="uV")
            sf = coupling[n_back].info['sfreq']

            win = int(4 * sf)  # Window size is set to 4 seconds

            concatenated_data.append(data)
            
        data_eeg = np.concatenate(concatenated_data, axis = 0) # shape is (epochs, channels, samples)
        
        # Compute the relative power
        freqs, psd = welch(data_eeg, sf, nperseg=win) 
        bp_relative = bandpower_from_psd_ndarray(psd, freqs, bands, ln_normalization = False, relative = True)

        p_bp_relative.append(bp_relative)
    
    coupling_p = list()
    for coupling in p_bp_relative:
        
        coupling = np.expand_dims(coupling, axis=0)
        #print("This is the shape of coupling for patients", coupling.shape)
        coupling_p.append(coupling)

    subjects_p.append(coupling_p)

delays_p = [list() for _ in range(10)]

for i in subjects_p:
    for j in range(10):
        delays_p[j].append(i[j])

delays_p = [np.concatenate(delay, axis=0) for delay in delays_p]

p_mean = list()
for delay in delays_p:
    #print("This is the shape", delay.shape)
    theta_bp_relative_p = delay[:, 1, : , :] #(subj, freq band, epochs, channels)
    mean_p = np.mean(theta_bp_relative_p, axis=0)
    #print("This is the shape of mean_p", mean_p.shape)
    mean_channels_p = np.mean(mean_p, axis=-1)
    #print("This is the shape of mean_channels_p", mean_channelsv.shape)
    mean_epochs_p = np.mean(mean_channels_p, axis=0)
    print("This is mean_epochs_p", mean_epochs_p)
    p_mean.append(mean_epochs_p)

print("This is the length of p_mean", len(p_mean))
print(len(delay_time))

plt.plot(delay_time, p_mean, label="patients")
plt.plot(delay_time, hc_mean, label = "healthy")
plt.xlabel("delay")
plt.ylabel("relative theta power")
plt.legend()
plt.show()

