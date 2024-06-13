import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
from scipy.signal import welch
import mne
from arrange_files import read_files

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

def process_subjects(file_dirs, epoch_duration, bands):
    subjects = []
    for file in file_dirs:
        raw = mne.io.read_raw_fif(file)
        raw.load_data()
        epochs_eeg = characterization_eeg(raw, epoch_duration)
        bp_relative_nback = []
        for nback in epochs_eeg:
            coupling_list = []
            for segment in nback:
                segment.drop_channels(['AF7', 'AFF5h', 'FT7', 'FC5', 'FC3', 'FCC3h', 'CCP3h', 'CCP1h', 'CP1', 
                                       'TP7', 'CPP3h', 'P1', 'AF8', 'AFF6h', 'FT8', 'FC6', 'FC4', 'FCC4h', 
                                       'CCP4h', 'CCP2h', 'CP2', 'P2', 'CPP4h', 'TP8'])
                data = segment.get_data(units="uV")
                sf = segment.info['sfreq']
                win = int(4 * sf)
                freqs, psd = welch(data, sf, nperseg=win)
                bp_relative = bandpower_from_psd_ndarray(psd, freqs, bands, ln_normalization=False, relative=True)
                coupling_list.append(bp_relative)
            bp_relative_nback.append(coupling_list)
        
        subject_data = []
        for n_back in bp_relative_nback:
            coupling_list_bp = [np.expand_dims(coupling, axis=0) for coupling in n_back]
            subject_data.append(coupling_list_bp)
        subjects.append(subject_data)
    
    return subjects

def concatenate_by_delay(subjects):
    concatenated_by_delay = {}
    for delay_index in range(len(subjects[0])):
        arrays_by_shape = {}
        for subject in subjects:
            for coupling in subject[delay_index]:
                shape = coupling.shape
                if shape not in arrays_by_shape:
                    arrays_by_shape[shape] = []
                arrays_by_shape[shape].append(coupling)
        concatenated_subject = {shape: np.concatenate(arrays, axis=0) for shape, arrays in arrays_by_shape.items()}
        concatenated_by_delay[delay_index] = concatenated_subject
    return concatenated_by_delay

def extract_delays(concatenated_by_delay):
    delays = [[] for _ in range(len(concatenated_by_delay))]
    for delay_index, subject in concatenated_by_delay.items():
        for shape, array in subject.items():
            delays[delay_index].append(array)
    return delays

def calculate_means(delays, label):
    means_back = []
    for delay in delays:
        all_means = []
        for sub_delay in delay:

            theta_bp_relative = sub_delay[:, 1, :, :]
            mean = np.mean(theta_bp_relative, axis=0)
            mean_channels = np.mean(mean, axis=-1)
            mean_epochs = np.mean(mean_channels, axis=0)
            print("This is the shape of mean epochs", mean_epochs.shape)
            all_means.append(mean_epochs) 

        means_back.append(all_means)

    print(f"This is the length of {label} mean: {len(means_back)}")

    return means_back

def plot_means(delay_time, hc_means, p_means):
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    titles = ["0back", "1back", "2back", "3back"]
    for i, (hc_mean, p_mean, title) in enumerate(zip(hc_means, p_means, titles)):
        ax = axs[i//2, i%2]
        ax.plot(delay_time, p_mean, label="patients")
        ax.plot(delay_time, hc_mean, label="healthy")
        ax.set_xlabel("delay")
        ax.set_ylabel("relative theta power")
        ax.set_title(title)
        ax.legend()
    plt.tight_layout()
    plt.show()

# Main Execution
os.chdir("/Users/adriana/Documents/GitHub/thesis")
mne.set_log_level('error')

clean_raw_eeg_hc = "/Users/adriana/Documents/DTU/thesis/data_acquisition/clean_eeg/healthy_controls/"
clean_raw_eeg_p = "/Users/adriana/Documents/DTU/thesis/data_acquisition/clean_eeg/patients/"

file_dirs_eeg_hc, _ = read_files(clean_raw_eeg_hc, '.fif')
file_dirs_eeg_p, _ = read_files(clean_raw_eeg_p, '.fif')

bands = [(0.5, 4, 'Delta'), (4, 8, 'Theta'), (8, 12, 'Alpha'), (12, 16, 'Sigma'), (16, 30, 'Beta')]
epoch_duration = 1
delay_time = np.arange(0, 10, epoch_duration)

subjects_hc = process_subjects(file_dirs_eeg_hc, epoch_duration, bands)
subjects_p = process_subjects(file_dirs_eeg_p, epoch_duration, bands)

concatenated_by_delay_hc = concatenate_by_delay(subjects_hc)
concatenated_by_delay_p = concatenate_by_delay(subjects_p)

delays_hc = extract_delays(concatenated_by_delay_hc)
delays_p = extract_delays(concatenated_by_delay_p)

hc_means = calculate_means(delays_hc, "hc_mean")
p_means = calculate_means(delays_p, "p_mean")

plot_means(delay_time, hc_means, p_means)