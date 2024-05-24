import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import mne
from arrange_files import read_files
from Hemo import HemoData

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
        print(f"This is the shape of delay: {delay.shape}")
        hbo_mean = delay[:, :, :, 0]
        mean_across_subjects = np.mean(hbo_mean, axis=0)
        mean_across_channels = np.mean(mean_across_subjects, axis=-1)
        mean_across_epochs = np.mean(mean_across_channels, axis=0)
        mean_values.append(mean_across_epochs)
        print(f"This is the shape of mean_values: {mean_across_epochs.shape}")
    return mean_values

os.chdir("/Users/adriana/Documents/GitHub/thesis")
mne.set_log_level('error')
mne.set_config('MNE_BROWSER_BACKEND', 'matplotlib')

clean_raw_fnirs_hc = "/Users/adriana/Documents/DTU/thesis/data_acquisition/data_fnirs/healthy_controls/"
clean_raw_fnirs_p = "/Users/adriana/Documents/DTU/thesis/data_acquisition/data_fnirs/patients/"

file_dirs_fnirs_hc, _ = read_files(clean_raw_fnirs_hc, '.snirf')
file_dirs_fnirs_p, _ = read_files(clean_raw_fnirs_p, '.snirf')

epoch_duration = 1
delay_time = np.arange(0, 10, epoch_duration)

hc_mean = process_fnirs_data(file_dirs_fnirs_hc, epoch_duration)
p_mean = process_fnirs_data(file_dirs_fnirs_p, epoch_duration)

print(f"This is the length of hc_mean: {len(hc_mean)}")
print(f"This is the length of p_mean: {len(p_mean)}")

plt.plot(delay_time, p_mean, label="patients")
plt.plot(delay_time, hc_mean, label="healthy")
plt.xlabel("delay")
plt.ylabel("hbo concentration")
plt.legend()
plt.show()