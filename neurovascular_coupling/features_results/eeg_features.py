import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import mne
from scipy.integrate import simps
from scipy.signal import welch
from autoreject import get_rejection_threshold
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from mne.viz import plot_alignment
from mne.viz import Brain

from basic.arrange_files import read_files
from neurovascular_coupling.Hemo import HemoData
#from arrange_files import read_files
#from Hemo import HemoData

"""

Characterize data

"""

# Function to visualize dropped epochs
def visualize_dropped_epochs(epochs_0back, epochs_1back, epochs_2back, epochs_3back):

    dropped_epochs_indices = []

    # Step 4: Retrieve the indices of the dropped epochs
    dropped_epochs_indices_0back = epochs_0back.drop_log
    dropped_epochs_indices_1back = epochs_1back.drop_log
    dropped_epochs_indices_2back = epochs_2back.drop_log
    dropped_epochs_indices_3back = epochs_3back.drop_log

    # Convert the drop_log to a list of indices of dropped epochs
    dropped_indices_0back = [i for i, log in enumerate(dropped_epochs_indices_0back) if len(log) > 0]
    dropped_indices_1back = [i for i, log in enumerate(dropped_epochs_indices_1back) if len(log) > 0]
    dropped_indices_2back = [i for i, log in enumerate(dropped_epochs_indices_2back) if len(log) > 0]
    dropped_indices_3back = [i for i, log in enumerate(dropped_epochs_indices_3back) if len(log) > 0]

    dropped_epochs_indices.append(dropped_indices_0back)
    dropped_epochs_indices.append(dropped_indices_1back)
    dropped_epochs_indices.append(dropped_indices_2back)
    dropped_epochs_indices.append(dropped_indices_3back)
    

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

    # Epoch data

    """
    Returns a mne.Epochs object created from the annotated mne.io.Raw object.

    """

    eeg_epochs = list()
    eeg_bad_epochs = list()

    events_0back = mne.make_fixed_length_events(raw_0back, start = 0 , stop = 58, duration = epoch_duration)
    events_1back = mne.make_fixed_length_events(raw_1back, start = 0 , stop = 70, duration = epoch_duration)
    events_2back = mne.make_fixed_length_events(raw_2back, start = 0 , stop = 70, duration = epoch_duration)
    events_3back = mne.make_fixed_length_events(raw_3back, start = 0 , stop = 60, duration = epoch_duration)
        
    interval = (0,0)
    epochs_0back = mne.Epochs(raw_0back, events_0back, baseline = interval, preload = True)
    epochs_1back = mne.Epochs(raw_1back, events_1back, baseline = interval, preload = True)
    epochs_2back = mne.Epochs(raw_2back, events_2back, baseline = interval, preload = True)
    epochs_3back = mne.Epochs(raw_3back, events_3back, baseline = interval, preload = True)
        
    # -----------------Get-Rejection-Threshold---------------------------------
    reject_thresholds_0back = get_rejection_threshold(epochs_0back, ch_types = "eeg", verbose = False)
    reject_thresholds_1back = get_rejection_threshold(epochs_1back, ch_types = "eeg", verbose = False)
    reject_thresholds_2back = get_rejection_threshold(epochs_2back, ch_types = "eeg", verbose = False)
    reject_thresholds_3back = get_rejection_threshold(epochs_3back, ch_types = "eeg", verbose = False)
        
    epochs_0back.drop_bad(reject=reject_thresholds_0back)
    epochs_1back.drop_bad(reject=reject_thresholds_1back)
    epochs_2back.drop_bad(reject=reject_thresholds_2back)
    epochs_3back.drop_bad(reject=reject_thresholds_3back)

    # Visualize dropped epochs
    bad_epoch = visualize_dropped_epochs(epochs_0back, epochs_1back, epochs_2back, epochs_3back)
        
    eeg_epochs.append(epochs_0back)
    eeg_epochs.append(epochs_1back)
    eeg_epochs.append(epochs_2back)
    eeg_epochs.append(epochs_3back)

    eeg_bad_epochs.append(bad_epoch)

    return (eeg_epochs, eeg_bad_epochs)

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

    # Epoch data


    fnirs_epochs = list()

    for bad in bad_epochs:
        #print("This is bad", bad)

        events_0back = mne.make_fixed_length_events(raw_0back, start = 0, stop= 58, duration = epoch_duration)
        events_1back = mne.make_fixed_length_events(raw_1back, start = 0, stop= 70, duration = epoch_duration)
        events_2back = mne.make_fixed_length_events(raw_2back, start = 0, stop= 70, duration = epoch_duration)
        events_3back = mne.make_fixed_length_events(raw_3back, start = 0, stop= 60, duration = epoch_duration)
        
        interval = (0,0)
        epochs_0back = mne.Epochs(raw_0back, events_0back, baseline = interval, preload = True)
        epochs_1back = mne.Epochs(raw_1back, events_1back, baseline = interval, preload = True)
        epochs_2back = mne.Epochs(raw_2back, events_2back, baseline = interval, preload = True)
        epochs_3back = mne.Epochs(raw_3back, events_3back, baseline = interval, preload = True)

        epochs_0back.drop(bad[0][0:-1])
        epochs_1back.drop(bad[1][0:-1])
        epochs_2back.drop(bad[2][0:-1])
        epochs_3back.drop(bad[3][0:-1])

        fnirs_epochs.append(epochs_0back)
        fnirs_epochs.append(epochs_1back)
        fnirs_epochs.append(epochs_2back)
        fnirs_epochs.append(epochs_3back)

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

def process_fnirs_data(file_dirs_fnirs, epoch_duration, bad_epochs):
    
    subjects_fnirs_0back = []
    subjects_fnirs_1back = []
    subjects_fnirs_2back = []
    subjects_fnirs_3back = []

    for file_dir, bad_epoch in zip(file_dirs_fnirs, bad_epochs):
        
        raw_haemo = HemoData(file_dir, preprocessing=True, isPloting=False).getMneIoRaw()
        
        epochs_fnirs = characterization_fNIRS(raw_haemo, epoch_duration, bad_epoch)

        features_fnirs = []

        for nback in epochs_fnirs:

            raw_hbo = nback.get_data(picks=["hbo"])

            mean_std_per_epoch_per_channel = np.concatenate([
                np.expand_dims(np.mean(raw_hbo, axis=-1), axis=-1),
                np.expand_dims(np.std(raw_hbo, axis=-1), axis=-1)], axis=-1)

            features_fnirs.append(mean_std_per_epoch_per_channel)
        
        subjects_fnirs_0back.append(features_fnirs[0])
        subjects_fnirs_1back.append(features_fnirs[1])
        subjects_fnirs_2back.append(features_fnirs[2])
        subjects_fnirs_3back.append(features_fnirs[3])

    return (subjects_fnirs_0back, subjects_fnirs_1back, subjects_fnirs_2back, subjects_fnirs_3back)

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
subjects_eeg_0back_hc = []
subjects_eeg_1back_hc = []
subjects_eeg_2back_hc = []
subjects_eeg_3back_hc = []

subj_hc_bad_epochs = []

for file_hc in file_dirs_eeg_hc:
    raw = mne.io.read_raw_fif(file_hc)
    raw.load_data()
    
    characterize_eeg_hc = characterization_eeg(raw, epoch_duration)

    epochs_hc = characterize_eeg_hc[0]
    bad_epochs_hc = characterize_eeg_hc[1]
        
    bp_relative_hc = []

    for nback_hc in epochs_hc:
        #channel_drop = [ 'AF7', 'AFF5h', 'FT7', 'FC5', 'FC3', 'FCC3h', 
        #            'CCP3h', 'CP1', 'TP7', 'CPP3h', 'P1', 
        #            'AF8', 'AFF6h', 'FT8', 'FC6', 'FC4', 'FCC4h', 'CCP4h', 
        #            'CP2', 'P2', 'CPP4h', 'TP8']
        
        #nback_hc.drop_channels(channel_drop)
        data_hc = nback_hc.get_data(units="uV")

        sf = nback_hc[0].info['sfreq']
        
        nperseg = 256
        noverlap = 0.5*256
        freqs_hc, psd_hc = welch(data_hc, sf, nperseg=256, noverlap=0.5*256)
        bp = bandpower_from_psd_ndarray(psd_hc, freqs_hc, bands, ln_normalization=False, relative=True)
        bp_relative_hc.append(bp)

    subjects_eeg_0back_hc.append(bp_relative_hc[0])
    subjects_eeg_1back_hc.append(bp_relative_hc[1])
    subjects_eeg_2back_hc.append(bp_relative_hc[2])
    subjects_eeg_3back_hc.append(bp_relative_hc[3])

    subj_hc_bad_epochs.append(bad_epochs_hc)

for subject_0back_hc in subjects_eeg_0back_hc:
    print("This is the shape of the 0back hc delay EEG", subject_0back_hc.shape)

for subject_1back_hc in subjects_eeg_1back_hc:
    print("This is the shape of the 1back hc delay EEG", subject_1back_hc.shape)

for subject_2back_hc in subjects_eeg_2back_hc:
    print("This is the shape of the 2back hc delay EEG", subject_2back_hc.shape)

for subject_3back_hc in subjects_eeg_3back_hc:
    print("This is the shape of the 3back hc delay EEG", subject_3back_hc.shape)

# PATIENTS

subjects_eeg_0back_p = []
subjects_eeg_1back_p = []
subjects_eeg_2back_p = []
subjects_eeg_3back_p = []

subj_p_bad_epochs = []

for file_p in file_dirs_eeg_p:
    raw = mne.io.read_raw_fif(file_p)
    raw.load_data()
    
    characterize_eeg_p = characterization_eeg(raw, epoch_duration)

    epochs_p = characterize_eeg_p[0]
    bad_epochs_p = characterize_eeg_p[1]
        
    bp_relative_p = []

    for nback_p in epochs_p:
        #channel_drop = [ 'AF7', 'AFF5h', 'FT7', 'FC5', 'FC3', 'FCC3h', 
        #            'CCP3h',  'CP1', 'TP7', 'CPP3h', 'P1', 
        #            'AF8', 'AFF6h', 'FT8', 'FC6', 'FC4', 'FCC4h', 'CCP4h', 
        #             'CP2', 'P2', 'CPP4h', 'TP8']
        
        #nback_p.drop_channels(channel_drop)
        data_p = nback_p.get_data(units="uV")

        sf = nback_p[0].info['sfreq']

        nperseg = 256
        noverlap = 0.5*256
        freqs_p, psd_p = welch(data_p, sf, nperseg=256, noverlap=0.5*256)
        bp = bandpower_from_psd_ndarray(psd_p, freqs_p, bands, ln_normalization=False, relative=True)
        bp_relative_p.append(bp)

    subjects_eeg_0back_p.append(bp_relative_p[0])
    subjects_eeg_1back_p.append(bp_relative_p[1])
    subjects_eeg_2back_p.append(bp_relative_p[2])
    subjects_eeg_3back_p.append(bp_relative_p[3])

    subj_p_bad_epochs.append(bad_epochs_p)

for subject_0back_p in subjects_eeg_0back_p:
    print("This is the shape of the 0back p delay EEG", subject_0back_p.shape)

for subject_1back_p in subjects_eeg_1back_p:
    print("This is the shape of the 1back p delay EEG", subject_1back_p.shape)

for subject_2back_p in subjects_eeg_2back_p:
    print("This is the shape of the 2back p delay EEG", subject_2back_p.shape)

for subject_3back_p in subjects_eeg_3back_p:
    print("This is the shape of the 3back p delay EEG", subject_3back_p.shape)

fnirs_hc = process_fnirs_data(file_dirs_fnirs_hc, epoch_duration, subj_hc_bad_epochs)
fnirs_0back_hc = fnirs_hc[0]
fnirs_1back_hc = fnirs_hc[1]
fnirs_2back_hc = fnirs_hc[2]
fnirs_3back_hc = fnirs_hc[3]

for subject_fnirs_0back_hc in fnirs_0back_hc:
    print("This is the shape of the 0back hc fnirs", subject_fnirs_0back_hc.shape)

for subject_fnirs_1back_hc in fnirs_1back_hc:
    print("This is the shape of the 1back hc fnirs", subject_fnirs_1back_hc.shape)

for subject_fnirs_2back_hc in fnirs_2back_hc:
    print("This is the shape of the 2back hc fnirs", subject_fnirs_2back_hc.shape)

for subject_fnirs_3back_hc in fnirs_3back_hc:
    print("This is the shape of the 3back hc fnirs", subject_fnirs_3back_hc.shape)

fnirs_p = process_fnirs_data(file_dirs_fnirs_p, epoch_duration, subj_p_bad_epochs)
fnirs_0back_p = fnirs_p[0]
fnirs_1back_p = fnirs_p[1]
fnirs_2back_p = fnirs_p[2]
fnirs_3back_p = fnirs_p[3]

for subject_fnirs_0back_p in fnirs_0back_p:
    print("This is the shape of the 0back p fnirs", subject_fnirs_0back_p.shape)

for subject_fnirs_1back_p in fnirs_1back_p:
    print("This is the shape of the 1back p fnirs", subject_fnirs_1back_p.shape)

for subject_fnirs_2back_p in fnirs_2back_p:
    print("This is the shape of the 2back p fnirs", subject_fnirs_2back_p.shape)

for subject_fnirs_3back_p in fnirs_3back_p:
    print("This is the shape of the 3back p fnirs", subject_fnirs_3back_p.shape)

from scipy.stats import pearsonr

def extract_bandpower_data(subjects_eeg):
    band_powers = {'delta': [], 'theta': []}
    #band_powers = {'delta': [], 'theta': [], 'alpha': [], 'beta': []}
    for subject in subjects_eeg:
        band_powers['delta'].append(np.mean(subject[0, :, :], axis=0))
        band_powers['theta'].append(np.mean(subject[1, :, :], axis=0))
        #band_powers['alpha'].append(np.mean(subject[2, :, :], axis=0))
        #band_powers['beta'].append(np.mean(subject[3, :, :], axis=0))
    return band_powers

def plot_topomaps(band_data, title, info, n_cols=5):
    n_subjects = len(band_data)
    n_rows = int(np.ceil(n_subjects / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
    if n_rows > 1:
        axes = axes.flatten()

    all_data = np.concatenate(band_data)
    vmin, vmax = all_data.min(), all_data.max()

    for idx, data in enumerate(band_data):
        evoked = mne.EvokedArray(data[:, np.newaxis], info)
        ax = axes[idx]
        im, _ = mne.viz.plot_topomap(evoked.data[:, 0], evoked.info, extrapolate = "box", cmap='Spectral_r', res = 32, contours=4, show=False, axes=ax, vlim= (vmin, vmax))
        ax.set_title(f'Subject {idx + 1}')

    for idx in range(n_subjects, len(axes)):
        axes[idx].axis('off')

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.show()


def eeg_features(subjects_eeg_0back, subjects_eeg_1back, subjects_eeg_2back, subjects_eeg_3back, title_suffix):
    nback_tasks = {
        '0-Back': subjects_eeg_0back,
        '1-Back': subjects_eeg_1back,
        '2-Back': subjects_eeg_2back,
        '3-Back': subjects_eeg_3back
    }
    
    channel_names = ["AFp1", "AFF1h", "FFC1h",  "FCC1h", "CCP1h",
                     "AFp2", "AFF2h", "FFC2h", "FCC2h", "CCP2h", 'AF7', 'AFF5h', 'FT7', 'FC5', 'FC3', 'FCC3h', 
                    'CCP3h', 'CP1', 'TP7', 'CPP3h', 'P1', 
                    'AF8', 'AFF6h', 'FT8', 'FC6', 'FC4', 'FCC4h', 'CCP4h', 
                    'CP2', 'P2', 'CPP4h', 'TP8']
    
    info = mne.create_info(ch_names=channel_names, sfreq=500, ch_types='eeg')
    montage = mne.channels.make_standard_montage("brainproducts-RNP-BA-128")
    info.set_montage(montage)

    for task, subjects_eeg in nback_tasks.items():
        band_powers = extract_bandpower_data(subjects_eeg)
        
        for band, data in band_powers.items():
            plot_topomaps(data, f'Topographic Maps for {band.capitalize()} Power in {task} Task - {title_suffix}', info)
            

# Example usage:
eeg_features(subjects_eeg_0back_hc, subjects_eeg_1back_hc, subjects_eeg_2back_hc, subjects_eeg_3back_hc, title_suffix='healthy controls')
eeg_features(subjects_eeg_0back_p, subjects_eeg_1back_p, subjects_eeg_2back_p, subjects_eeg_3back_p, title_suffix='patients')