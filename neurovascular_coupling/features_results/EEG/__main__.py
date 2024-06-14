import os
import numpy as np
import matplotlib.pyplot as plt
import mne
from scipy.integrate import simps
from scipy.signal import welch
from autoreject import get_rejection_threshold
from scipy.stats import ttest_1samp
import seaborn as sns
sns.set(font_scale=1.2)

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
        epochs_0back.drop(bad[0][0:-1])
        #epochs_baseline_1back.drop(bad[2][0:-1])
        epochs_1back.drop(bad[1][0:-1])
        #epochs_baseline_2back.drop(bad[4][0:-1])
        epochs_2back.drop(bad[2][0:-1])
        #epochs_baseline_3back.drop(bad[6][0:-1])
        epochs_3back.drop(bad[3][0:-1])

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

"""

Process data

"""

# Function to extract the mean HBO concentration from fNIRS epochs
def extract_fnirs_features(epochs_fnirs):
    features_fnirs = []
    for epoch in epochs_fnirs:
        raw_hbo = epoch.get_data(picks=["hbo"])
        mean_std_per_epoch_per_channel = np.concatenate([
            np.expand_dims(np.mean(raw_hbo, axis=-1), axis=-1),
            np.expand_dims(np.std(raw_hbo, axis=-1), axis=-1)], axis=-1)
        features_fnirs.append(mean_std_per_epoch_per_channel)
    return features_fnirs

# Function to subtract baseline from n-back tasks
def subtract_baseline_fnirs(nback_epochs, baseline_epochs):
    for i in range(len(nback_epochs)):
        baseline_data = baseline_epochs[i].get_data()
        nback_data = nback_epochs[i].get_data()
        baseline_mean = np.mean(baseline_data, axis=0, keepdims=True)
        nback_epochs[i]._data -= baseline_mean
    return nback_epochs

# Function to process fNIRS data
def process_fnirs_data(file_dirs_fnirs, epoch_duration, bad_epochs):
    results = { "nback": [[], [], [], []], "baseline": [[], [], [], []] }

    for file_dir, bad_epoch in zip(file_dirs_fnirs, bad_epochs):
        raw_haemo = HemoData(file_dir, preprocessing=True, isPloting=False).getMneIoRaw()

        epochs_fnirs = characterization_fNIRS(raw_haemo, epoch_duration, bad_epoch)

        nback_epochs = epochs_fnirs[0]
        baseline_epochs = epochs_fnirs[1]

        # Subtract baseline from each n-back task
        nback_epochs = subtract_baseline_fnirs(nback_epochs, baseline_epochs)

        features_nback = extract_fnirs_features(nback_epochs)
        features_baseline = extract_fnirs_features(baseline_epochs)

        for i in range(len(nback_epochs)):
            results["nback"][i].append(features_nback[i])
            results["baseline"][i].append(features_baseline[i])

    return results

# Function to extract the Relative Bandpower
def extract_bandpower_from_epochs(epochs, sf, bands, plot=False):
    bp_relative = []

    for nback in epochs:
        # channel_names = nback.ch_names
        
        data = nback.get_data(units="uV")
 
        nperseg = 4 * sf
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
            plt.ylim([0, psd_mean[0,:].max()*1.1])
            # plt.title(channel_names[0])  # Uncomment if you want to use channel names
            plt.xlim([0, freqs.max()])
            plt.xlim([0, 50])
            sns.despine()
            plt.show()

        bp = bandpower_from_psd_ndarray(psd, freqs, bands, ln_normalization=True, relative=True)
        bp_relative.append(bp)

    return bp_relative

# Function to subtract baseline from n-back tasks
def subtract_baseline_eeg(nback_epochs, baseline_epochs):
    for nback_epoch, baseline_epoch in zip(nback_epochs, baseline_epochs):
        nback_data = nback_epoch.get_data()
        baseline_mean = baseline_epoch.get_data().mean(axis=0, keepdims=True)
        nback_epoch._data -= baseline_mean
    return nback_epochs

# Function to process EEG data
def process_eeg_data(file_dirs, epoch_duration, bands):
    
    results = { "nback": [[], [], [], []], "baseline": [[], [], [], []] }
    
    subj_bad_epochs = []

    for file in file_dirs:
        raw = mne.io.read_raw_fif(file, preload=True)
        
        # Channels to drop
        channels_to_drop = ["AF7", "AFF5h", "AF8", "AFF6h", "FT7", "FC5", "FC3", "FCC3h", "CCP3h",
                            "CP1", "CPP3h", "P1", "TP7", "CCP1h", "CCP2h", "CP2", "CPP4h", "P2", "TP8", 
                            "CCP4h", "FCC4h", "FC4", "FC6", "FT8", "AFF6h", "AF8", "FCC1h", "FCC2h"]
        
        # Drop specified channels
        raw.drop_channels(channels_to_drop)
        #raw.plot_sensors(show_names=True, block=True)

        # Apply a notch filter 
        raw.filter(l_freq = 1, h_freq = 35, verbose = False)
        #raw.notch_filter(50, verbose=False)

        raw.load_data()

        epochs, baseline_epochs, bad_epochs = characterization_eeg(raw, epoch_duration)
        sf = raw.info['sfreq']
        
        # Subtract baseline from each n-back task
        epochs = subtract_baseline_eeg(epochs, baseline_epochs)
        
        bp_nback = extract_bandpower_from_epochs(epochs, sf, bands, plot=False)
        bp_baseline = extract_bandpower_from_epochs(baseline_epochs, sf, bands, plot=False)

        for i, (bp_n, bp_b) in enumerate(zip(bp_nback, bp_baseline)):
            results["nback"][i].append(bp_n)
            results["baseline"][i].append(bp_b)

        subj_bad_epochs.append(bad_epochs)

    return results, subj_bad_epochs

# Apply one sample t test to EEG data
def one_sample_ttest_eeg(data):
    # Collect mean_channels from each array
    mean_channels_list = []

    for array in data:
        theta = array[1, :, :]
        mean_channels = np.mean(theta, axis=-1)
        mean_channels_list.append(mean_channels)
    
    # Find the maximum length among all mean_channels
    max_length = max(len(arr) for arr in mean_channels_list)

    # Pad all mean_channels to the maximum length with NaNs
    padded_mean_channels = np.full((len(mean_channels_list), max_length), np.nan)

    for i, arr in enumerate(mean_channels_list):
        padded_mean_channels[i, :len(arr)] = arr

    # Compute the mean while ignoring NaNs
    overall_mean = np.nanmean(padded_mean_channels, axis=0)
    
    # Now compute the mean of the overall_mean to get the mean theta power
    mean_theta_power = np.nanmean(overall_mean)
    #print("Mean theta power:", mean_theta_power)

    t_stat, p_value = ttest_1samp(overall_mean, popmean=0, axis=0)

    return t_stat, p_value, mean_theta_power

# Apply one sample t test to fNIRS data
def one_sample_ttest_fnirs(data):
    # Collect mean_channels from each array
    mean_channels_list = []

    for array in data:
        hbo = array[:, 0:4, 0]
        mean_channels = np.mean(hbo, axis=-1)
        mean_channels_list.append(mean_channels)
    
    # Find the maximum length among all mean_channels
    max_length = max(len(arr) for arr in mean_channels_list)

    # Pad all mean_channels to the maximum length with NaNs
    padded_mean_channels = np.full((len(mean_channels_list), max_length), np.nan)

    for i, arr in enumerate(mean_channels_list):
        padded_mean_channels[i, :len(arr)] = arr

    # Compute the mean while ignoring NaNs
    overall_mean = np.nanmean(padded_mean_channels, axis=0)
    
    # Now compute the mean of the overall_mean to get the mean theta power
    mean_hbo = np.nanmean(overall_mean)

    t_stat, p_value = ttest_1samp(overall_mean, popmean=0, axis=0)

    return t_stat, p_value, mean_hbo

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
    #plt.show()

def eeg_features(subjects_eeg_0back, subjects_eeg_1back, subjects_eeg_2back, subjects_eeg_3back, title_suffix):
    nback_tasks = {
        '0-Back': subjects_eeg_0back,
        '1-Back': subjects_eeg_1back,
        '2-Back': subjects_eeg_2back,
        '3-Back': subjects_eeg_3back
    }
    
    channel_names = ["AFp1", "AFF1h", "FFC1h",   
                     "AFp2", "AFF2h", "FFC2h"]
    
    info = mne.create_info(ch_names=channel_names, sfreq=500, ch_types='eeg')
    montage = mne.channels.make_standard_montage("brainproducts-RNP-BA-128")
    info.set_montage(montage)

    for task, subjects_eeg in nback_tasks.items():
        band_powers = extract_bandpower_data(subjects_eeg)
        
        for band, data in band_powers.items():
            plot_topomaps(data, f'Topographic Maps for {band.capitalize()} Power in {task} Task - {title_suffix}', info)

def plot_mean_theta(eeg_results, title_suffix):
    
    # Collect mean_channels from each array
    mean_channels_list = []

    for array in eeg_results:
        theta = array[1, :, :]
        mean_channels = np.mean(theta, axis=-1)
        mean_channels_list.append(mean_channels)
    
    # Find the maximum length among all mean_channels
    max_length = max(len(arr) for arr in mean_channels_list)

    # Pad all mean_channels to the maximum length with NaNs
    padded_mean_channels = np.full((len(mean_channels_list), max_length), np.nan)

    for i, arr in enumerate(mean_channels_list):
        padded_mean_channels[i, :len(arr)] = arr

    # Compute the mean while ignoring NaNs
    overall_mean = np.nanmean(padded_mean_channels, axis=0)
    overall_std = np.nanstd(padded_mean_channels, axis=0)

    return overall_mean, overall_std


# Function to plot mean HBO concentration
def plot_mean_hbo(fnirs_results, title_suffix):
    
    subjects_list = []
    for array in fnirs_results:
        mean_hbo_data = np.mean(np.array(array)[:, :, 0], axis=0)
        subjects_list.append(mean_hbo_data)

    # Find the maximum length among all mean_channels
    max_length = max(len(arr) for arr in subjects_list)

    # Pad all mean_channels to the maximum length with NaNs
    padded_mean_subjects = np.full((len(subjects_list), max_length), np.nan)

    for i, arr in enumerate(subjects_list):
        padded_mean_subjects[i, :len(arr)] = arr

    # Compute the mean while ignoring NaNs
    overall_mean = np.nanmean(padded_mean_subjects, axis=0)
    overall_std = np.nanstd(padded_mean_subjects, axis=0)
    
    print("this is the shape of overall_mean", overall_mean.shape)
    return overall_mean, overall_std

if __name__ == "__main__":
    def main():
        os.chdir("/Users/adriana/Documents/GitHub/thesis")
        mne.set_log_level('error')
        mne.set_config('MNE_BROWSER_BACKEND', 'matplotlib')

        clean_raw_eeg_hc = "/Users/adriana/Documents/DTU/thesis/data_acquisition/clean_eeg/healthy_controls/"
        clean_raw_eeg_p = "/Users/adriana/Documents/DTU/thesis/data_acquisition/clean_eeg/patients/"
        clean_raw_fnirs_hc = "/Users/adriana/Documents/DTU/thesis/data_acquisition/data_fnirs/healthy_controls/"
        clean_raw_fnirs_p = "/Users/adriana/Documents/DTU/thesis/data_acquisition/data_fnirs/patients/"

        file_dirs_eeg_hc, _ = read_files(clean_raw_eeg_hc, '.fif')
        file_dirs_eeg_p, _ = read_files(clean_raw_eeg_p, '.fif')
        file_dirs_fnirs_hc, _ = read_files(clean_raw_fnirs_hc, '.snirf')
        file_dirs_fnirs_p, _ = read_files(clean_raw_fnirs_p, '.snirf')

        bands = [(0.5, 4, 'Delta'), (4, 8, 'Theta'), (8, 12, 'Alpha'), 
                 (12, 16, 'Sigma'), (16, 30, 'Beta')]
        
        epoch_duration = 5

        # Process EEG data
        eeg_hc_results, subj_hc_bad_epochs = process_eeg_data(file_dirs_eeg_hc, epoch_duration, bands)
        eeg_p_results, subj_p_bad_epochs = process_eeg_data(file_dirs_eeg_p, epoch_duration, bands)

        # Process fNIRS data
        fnirs_hc_results = process_fnirs_data(file_dirs_fnirs_hc, epoch_duration, subj_hc_bad_epochs)
        fnirs_p_results = process_fnirs_data(file_dirs_fnirs_p, epoch_duration, subj_p_bad_epochs)

        """
        
        EEG
        
        """

        # Apply one-sample t-test to EEG data
        for i in range(4):
            t_stat_hc, p_value_hc, mean_theta_power_hc = one_sample_ttest_eeg(eeg_hc_results["nback"][i])
            t_stat_p, p_value_p, mean_theta_power_p = one_sample_ttest_eeg(eeg_p_results["nback"][i])
            print(f"{i}-back Healthy Controls Theta power: {mean_theta_power_hc}")
            print(f"{i}-back Patients Theta power: {mean_theta_power_p}")
            print(f"{i}-back Healthy Controls t-statistic: {t_stat_hc}, p-value: {p_value_hc}")
            print(f"{i}-back Patients t-statistic: {t_stat_p}, p-value: {p_value_p}")
        
        # Create combined boxplot for the mean HBO concentration data for healthy controls and patients
        plt.figure(figsize=(10, 6))
        mean_theta_data_hc = []
        mean_theta_data_p = []
        std_theta_data_hc = []
        std_theta_data_p = []

        for i in range(4):
            theta_channel_mean_hc, _ = plot_mean_theta(eeg_hc_results["nback"][i], title_suffix='healthy controls')
            theta_channel_mean_p, _ = plot_mean_theta(eeg_p_results["nback"][i], title_suffix='patients')
            mean_theta_data_hc.append(theta_channel_mean_hc)
            mean_theta_data_p.append(theta_channel_mean_p)
            _, theta_channel_std_hc = plot_mean_theta(eeg_hc_results["nback"][i], title_suffix='healthy controls')
            _, theta_channel_std_p = plot_mean_theta(eeg_p_results["nback"][i], title_suffix='patients')
            std_theta_data_hc.append(theta_channel_std_hc)
            std_theta_data_p.append(theta_channel_std_p)

        positions = np.arange(1, 9)

        # Create a combined list for the labels
        labels = ['HC 0-Back', 'P 0-Back', 'HC 1-Back', 'P 1-Back', 'HC 2-Back', 'P 2-Back', 'HC 3-Back', 'P 3-Back']

        # Plot boxplot for healthy controls
        plt.boxplot(mean_theta_data_hc, positions=positions[::2], widths=0.4, patch_artist=True, boxprops=dict(facecolor="orange"))
        # Plot boxplot for patients
        plt.boxplot(mean_theta_data_p, positions=positions[1::2], widths=0.4, patch_artist=True, boxprops=dict(facecolor="green"))

        # Add error bars
        for i in range(4):
            plt.errorbar([positions[i * 2]] * len(mean_theta_data_hc[i]), mean_theta_data_hc[i], yerr=std_theta_data_hc[i], fmt='o', color='orange')
            plt.errorbar([positions[i * 2 + 1]] * len(mean_theta_data_p[i]), mean_theta_data_p[i], yerr=std_theta_data_p[i], fmt='o', color='green')

        plt.xlabel('N-Back Task and Group')
        plt.ylabel('Theta Power')
        plt.title('Boxplot of Theta Power for N-Back Tasks')
        plt.xticks(positions, labels)
        plt.show()

        """
        
        fNIRS
        
        """

        # Apply one-sample t-test to fNIRS data
        for i in range(4):
            t_stat_hc, p_value_hc, mean_hbo_hc = one_sample_ttest_fnirs(fnirs_hc_results["nback"][i])
            t_stat_p, p_value_p, mean_hbo_p = one_sample_ttest_fnirs(fnirs_p_results["nback"][i])
            print(f"{i}-back Healthy Controls HBO: {mean_hbo_hc}")
            print(f"{i}-back Patients HBO: {mean_hbo_p}")
            print(f"{i}-back Healthy Controls HBO t-statistic: {t_stat_hc}, p-value: {p_value_hc}")
            print(f"{i}-back Patients HBO t-statistic: {t_stat_p}, p-value: {p_value_p}")
        
        # Plot mean HBO concentration for healthy controls and patients
        plt.figure(figsize=(14, 6))
        for i in range(4):
            plt.subplot(2, 2, i + 1)
            
            # Healthy Controls
            hbo_channel_mean_hc, hbo_channel_std_hc = plot_mean_hbo(fnirs_hc_results["nback"][i], title_suffix='healthy controls')
            plt.plot(hbo_channel_mean_hc, label=f'HC {i}-Back Task', color='blue')
            plt.fill_between(range(len(hbo_channel_mean_hc)), hbo_channel_mean_hc - hbo_channel_std_hc, hbo_channel_mean_hc + hbo_channel_std_hc, alpha=0.2, color='blue')

            # Patients
            hbo_channel_mean_p, hbo_channel_std_p = plot_mean_hbo(fnirs_p_results["nback"][i], title_suffix='patients')
            plt.plot(hbo_channel_mean_p, label=f'Patient {i}-Back Task', color='red')
            plt.fill_between(range(len(hbo_channel_mean_p)), hbo_channel_mean_p - hbo_channel_std_p, hbo_channel_mean_p + hbo_channel_std_p, alpha=0.2, color='red')

            plt.xlabel('Channels')
            plt.ylabel('Mean HBO Concentration')
            plt.title(f'{i}-Back Task HBO Concentration')
            plt.legend()

        plt.tight_layout()
        plt.show()

        # Create combined boxplot for the mean HBO concentration data for healthy controls and patients
        plt.figure(figsize=(10, 6))
        
        mean_hbo_data_hc = []
        mean_hbo_data_p = []
        std_hbo_data_hc = []
        std_hbo_data_p = []

        for i in range(4):
            hbo_channel_mean_hc, _ = plot_mean_hbo(fnirs_hc_results["nback"][i], title_suffix='healthy controls')
            hbo_channel_mean_p, _ = plot_mean_hbo(fnirs_p_results["nback"][i], title_suffix='patients')
            mean_hbo_data_hc.append(hbo_channel_mean_hc)
            mean_hbo_data_p.append(hbo_channel_mean_p)
            _, hbo_channel_std_hc = plot_mean_hbo(fnirs_hc_results["nback"][i], title_suffix='healthy controls')
            _, hbo_channel_std_p = plot_mean_hbo(fnirs_p_results["nback"][i], title_suffix='patients')
            std_hbo_data_hc.append(hbo_channel_std_hc)
            std_hbo_data_p.append(hbo_channel_std_p)

        positions = np.arange(1, 9)

        # Create a combined list for the labels
        labels = ['HC 0-Back', 'P 0-Back', 'HC 1-Back', 'P 1-Back', 'HC 2-Back', 'P 2-Back', 'HC 3-Back', 'P 3-Back']

        # Plot boxplot for healthy controls
        plt.boxplot(mean_hbo_data_hc, positions=positions[::2], widths=0.4, patch_artist=True, boxprops=dict(facecolor="blue"))
        # Plot boxplot for patients
        plt.boxplot(mean_hbo_data_p, positions=positions[1::2], widths=0.4, patch_artist=True, boxprops=dict(facecolor="red"))

        # Add error bars
        for i in range(4):
            plt.errorbar([positions[i * 2]] * len(mean_hbo_data_hc[i]), mean_hbo_data_hc[i], yerr=std_hbo_data_hc[i], fmt='o', color='blue')
            plt.errorbar([positions[i * 2 + 1]] * len(mean_hbo_data_p[i]), mean_hbo_data_p[i], yerr=std_hbo_data_p[i], fmt='o', color='red')

        plt.xlabel('N-Back Task and Group')
        plt.ylabel('HBO Concentration')
        plt.title('Boxplot of HBO Concentration for N-Back Tasks')
        plt.xticks(positions, labels)
        plt.show()

        eeg_features(eeg_hc_results["nback"][0], eeg_hc_results["nback"][1], eeg_hc_results["nback"][2], eeg_hc_results["nback"][3], title_suffix='healthy controls')
        eeg_features(eeg_p_results["nback"][0], eeg_p_results["nback"][1], eeg_p_results["nback"][2], eeg_p_results["nback"][3], title_suffix='patients')

    main()