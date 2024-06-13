import mne
import numpy as np
import matplotlib.pyplot as plt
import os
from autoreject import AutoReject,get_rejection_threshold

from basic.arrange_files import read_files

def characterization_trigger_data(raw):
    
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
    # period (15s), and the test time (48s) & instructions (4s) of 1-Back Test
    # we add 10 seconds to start the segmentation 10 seconds after the trigger

    begin_trigger4 = trigger_data["5"]["begin"][0] - 15 - 48 - 4
    trigger_data["4"]["begin"].append(begin_trigger4)
    trigger_data["4"]["duration"].append(48) # Duration of 0-Back Test is 48 s

    end_time = trigger_data["4"]["begin"][0] + 48
    trigger_data["4"]["end"].append(end_time)

    #----------------------------------------------------------------------------------

    #----------------------------------------------------------------------------------

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
    
    raw_new=raw.set_annotations(new_annotations)
    #raw_new.plot(block=True)
    #plt.show()
    
    #----------------------------------------------------------------------------------

    raw_0back = raw_new.copy().crop(tmin=trigger_data['4']['begin'][0], tmax=trigger_data['4']['end'][0])
    raw_1back = raw_new.copy().crop(tmin=trigger_data['5']['begin'][0], tmax=trigger_data['5']['end'][0])
    raw_2back = raw_new.copy().crop(tmin=trigger_data['6']['begin'][0], tmax=trigger_data['6']['end'][0])
    raw_3back = raw_new.copy().crop(tmin=trigger_data['7']['begin'][0], tmax=trigger_data['7']['end'][0])
    

    return (raw_0back, raw_1back, raw_2back, raw_3back)

def epochs_from_raw(raw, raw_0back, raw_1back, raw_2back, raw_3back):
    """
    Returns a mne.Epochs object created from the annotated mne.io.Raw object.

    """
    # Create epochs of 1 s for all raw segments 
    events_0back = mne.make_fixed_length_events(raw_0back, start = 10, stop = 40, duration = 1)
    events_1back = mne.make_fixed_length_events(raw_1back, start = 10, stop = 52, duration = 1)
    events_2back = mne.make_fixed_length_events(raw_2back, start = 10, stop = 52, duration = 1)
    events_3back = mne.make_fixed_length_events(raw_3back, start = 10, stop = 52, duration = 1)

    interval = (0,0)
    epochs_0back = mne.Epochs(raw_0back, events_0back, baseline=interval, preload=True)
    epochs_1back = mne.Epochs(raw_1back, events_1back, baseline=interval, preload=True)
    epochs_2back = mne.Epochs(raw_2back, events_2back, baseline=interval, preload=True)
    epochs_3back = mne.Epochs(raw_3back, events_3back, baseline=interval, preload=True)
    
    # -----------------Get-Rejection-Threshold---------------------------------
    reject = get_rejection_threshold(epochs_0back, ch_types = "eeg")
    epochs_0back.drop_bad(reject=reject)

    reject = get_rejection_threshold(epochs_1back, ch_types = "eeg")
    epochs_1back.drop_bad(reject=reject)

    reject = get_rejection_threshold(epochs_2back, ch_types = "eeg")
    epochs_2back.drop_bad(reject=reject)

    reject = get_rejection_threshold(epochs_3back, ch_types = "eeg")
    epochs_3back.drop_bad(reject=reject)

    # -----------------AutoReject---------------------------------
    #n_interpolates = np.array([1,4,32])
    #consensus_percs = np.linspace(0,1.0,11)

    #ar = AutoReject(n_interpolates, consensus_percs, cv=5, thresh_method = "random_search", random_state = 42)

    #epochs_0back_clean, reject_log_0back = ar.fit_transform(epochs_0back, return_log = True)
    #epochs_1back_clean, reject_log_1back = ar.fit_transform(epochs_1back, return_log = True)
    #epochs_2back_clean, reject_log_2back = ar.fit_transform(epochs_2back, return_log = True)
    #epochs_3back_clean, reject_log_3back = ar.fit_transform(epochs_3back, return_log = True)

    return (epochs_0back, epochs_1back, epochs_2back, epochs_3back)

# Set default directory
os.chdir("H:\Dokumenter\GitHub\MasterThesis\.venv")
mne.set_log_level('error')

# Folder where to get the raw EEG files
clean_raw_folder = "H:\\Dokumenter\\data_acquisition\\data_eeg\\clean_eeg\\patients\\"
#clean_raw_folder =  "/Users/adriana/Documents/DTU/thesis/data_acquisition/clean_eeg/healthy_controls/"

# Folder where to export the clean epochs files
clean_epoch_folder =  "H:\\Dokumenter\\data_acquisition\\data_eeg\\clean_epochs\\patients\\"
#clean_epoch_folder = "/Users/adriana/Documents/DTU/thesis/data_acquisition/clean_eeg_epoch/healthy_controls/"

n_back = "n_back"

# Get directories of raw EEG files and set export directory for clean files
dir_inprogress = os.path.join(clean_raw_folder)
export_dir = os.path.join(clean_epoch_folder)

file_dirs, subject_names = read_files(dir_inprogress,'.fif')

mne.set_config('MNE_BROWSER_BACKEND', 'matplotlib')

# Loop through all the subjects' directories (EEG files directories)
for i in range(len(file_dirs)):
    
    # --------------Read Raw data + Bad channel detection + Filter-------------------
    raw = mne.io.read_raw_fif(file_dirs[i])
    raw.load_data()

    #raw.plot(block=True)
    #plt.show()
        
    raw_characterization = characterization_trigger_data(raw)

    raw_0back = raw_characterization[0]
    raw_1back = raw_characterization[1]
    raw_2back = raw_characterization[2]
    raw_3back = raw_characterization[3]

    # --------------Epoched Data--------------------------------
    # Each epoched object contains 1s epoched data corresponding to each data segment

    epochs=epochs_from_raw(raw, raw_0back, raw_1back, raw_2back, raw_3back)
    
    # Save the cleaned EEG file as .fif file
    try:
        os.makedirs(export_dir)
    except FileExistsError:
        pass
    
    try:
        count=0
        for j in epochs:
        # --------------Save as file--------------------------------
            if count<=3:
                fname = '{}/{}/{}_{}back_clean-epo.fif'.format(export_dir, n_back, subject_names[i], count)
                j.save(fname, overwrite=True)
                count = count+1

    except FileExistsError:
        pass




