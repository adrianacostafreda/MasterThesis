import numpy as np
import mne
from autoreject import AutoReject,get_rejection_threshold
from DataPath import DataPath
from Hemo import HemoData
from sklearn.preprocessing import StandardScaler

def invertDic(dic: dict):
        '''
        Helping function to invert the keys and the values of a dictionary.
        '''
        inv_dic = {v: k for k, v in dic.items()}
        return inv_dic

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
            begin_trigger = annot.onset[idx] + 11 # segment the data 10 seconds after the trigger
            duration_trigger = annot.duration[idx] + 50 # Durations of the test are 20 seconds
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
    begin_trigger4 = trigger_data["5"]["begin"][0] - 15 - 48 - 8 + 11
    trigger_data["4"]["begin"].append(begin_trigger4)
    trigger_data["4"]["duration"].append(37) # Duration of 0-Back Test is 48 s

    end_time = trigger_data["4"]["begin"][0] + 37
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
    tstep = 1
    events = mne.make_fixed_length_events(raw, duration=tstep)
    
    epochs_0back = mne.Epochs(raw_0back, events, tmin=0, tmax=tstep, baseline=(0, 0), preload=True)
    epochs_1back = mne.Epochs(raw_1back, events, tmin=0, tmax=tstep, baseline=(0, 0), preload=True)
    epochs_2back = mne.Epochs(raw_2back, events, tmin=0, tmax=tstep, baseline=(0, 0), preload=True)
    epochs_3back = mne.Epochs(raw_3back, events, tmin=0, tmax=tstep, baseline=(0, 0), preload=True)

    return (epochs_0back, epochs_1back, epochs_2back, epochs_3back)

#path = "/Users/adriana/Documents/DTU/thesis/data_acquisition/fNIRS/healthy_controls/"
path = "H:\\Dokumenter\\data_acquisition\\data_fnirs\\healthy_controls\\baseline\\snirf_files\\"

datapath = DataPath(path, recursive=False)
print(len(datapath.getDataPaths()))

MAX_LENGTH_HEALTHY = 204

bad_epochs = list()

# Loop over all files preprocess them with HemoData and save them at an numpy array.
# It save the data for each patient, for each event, for each channel:  patient X events X channels X time_samples.
for id,file in enumerate(datapath.getDataPaths()):
        raw_haemo = HemoData(file, preprocessing=True, isPloting=False).getMneIoRaw()
        #raw_haemo.plot(block=True)

        raw_characterization = characterization_trigger_data(raw_haemo)

        raw_0back = raw_characterization[0]
        raw_1back = raw_characterization[1]
        raw_2back = raw_characterization[2]
        raw_3back = raw_characterization[3]

        # --------------Epoched Data--------------------------------
        # Each epoched object contains 1s epoched data corresponding to each data segment

        epochs=epochs_from_raw(raw_haemo, raw_0back, raw_1back, raw_2back, raw_3back)

        bad_epochs.append(epochs[4])

        raw_data_hbo_0back = epochs[0].get_data(picks=["hbo"]).mean(0) # channels x samples
        raw_data_hbr_0back = epochs[0].get_data(picks=["hbr"]).mean(0) # channels x samples

        raw_data_hbo_1back = epochs[1].get_data(picks=["hbo"]).mean(0) # channels x samples
        raw_data_hbr_1back = epochs[1].get_data(picks=["hbr"]).mean(0) # channels x samples

        raw_data_hbo_2back = epochs[2].get_data(picks=["hbo"]).mean(0) # channels x samples
        raw_data_hbr_2back = epochs[2].get_data(picks=["hbr"]).mean(0) # channels x samples

        raw_data_hbo_3back = epochs[3].get_data(picks=["hbo"]).mean(0) # channels x samples
        raw_data_hbr_3back = epochs[3].get_data(picks=["hbr"]).mean(0) # channels x samples

        print("This is the shape of raw_data_hbo_0back", raw_data_hbr_0back.shape)

        # Standardize
        ss = StandardScaler()
        raw_data_hbo_0back = ss.fit_transform(raw_data_hbo_0back.T).T
        # The shape of raw data is (n_channels, samples)
        raw_data_hbr_0back = ss.fit_transform(raw_data_hbr_0back.T).T

        raw_data_hbo_1back = ss.fit_transform(raw_data_hbo_1back.T).T
        # The shape of raw data is (n_channels, samples)
        raw_data_hbr_1back = ss.fit_transform(raw_data_hbr_1back.T).T

        raw_data_hbo_2back = ss.fit_transform(raw_data_hbo_2back.T).T
        # The shape of raw data is (n_channels, samples)
        raw_data_hbr_2back = ss.fit_transform(raw_data_hbr_2back.T).T

        raw_data_hbo_3back = ss.fit_transform(raw_data_hbo_3back.T).T
        # The shape of raw data is (n_channels, samples)
        raw_data_hbr_3back = ss.fit_transform(raw_data_hbr_3back.T).T

        # We will expand each task 15 s which corresponds to 153
        t_interval = 153

        # Simple version subjects x events x channels x samples.
        if id == 0:
            data_hbo_0back = np.expand_dims(raw_data_hbo_0back[:,:],axis=0)
            data_hbr_0back = np.expand_dims(raw_data_hbr_0back[:,:],axis=0)
        else:
            data_hbo_0back = np.concatenate((data_hbo_0back, np.expand_dims(raw_data_hbo_0back[:,:],axis=0)),axis=0)
            data_hbr_0back = np.concatenate((data_hbr_0back, np.expand_dims(raw_data_hbr_0back[:,:],axis=0)),axis=0)

        # Simple version subjects x events x channels x samples.
        if id == 0:
            data_hbo_1back = np.expand_dims(raw_data_hbo_1back[:,:],axis=0)
            data_hbr_1back = np.expand_dims(raw_data_hbr_1back[:,:],axis=0)
        else:
            data_hbo_1back = np.concatenate((data_hbo_1back, np.expand_dims(raw_data_hbo_1back[:,:],axis=0)),axis=0)
            data_hbr_1back = np.concatenate((data_hbr_1back, np.expand_dims(raw_data_hbr_1back[:,:],axis=0)),axis=0)

        # Simple version subjects x events x channels x samples.
        if id == 0:
            data_hbo_2back = np.expand_dims(raw_data_hbo_2back[:,:],axis=0)
            data_hbr_2back = np.expand_dims(raw_data_hbr_2back[:,:],axis=0)
        else:
            data_hbo_2back = np.concatenate((data_hbo_2back, np.expand_dims(raw_data_hbo_2back[:,:],axis=0)),axis=0)
            data_hbr_2back = np.concatenate((data_hbr_2back, np.expand_dims(raw_data_hbr_2back[:,:],axis=0)),axis=0)

        # Simple version subjects x events x channels x samples.
        if id == 0:
            data_hbo_3back = np.expand_dims(raw_data_hbo_3back[:,:],axis=0)
            data_hbr_3back = np.expand_dims(raw_data_hbr_3back[:,:],axis=0)
        else:
            data_hbo_3back = np.concatenate((data_hbo_3back, np.expand_dims(raw_data_hbo_3back[:,:],axis=0)),axis=0)
            data_hbr_3back = np.concatenate((data_hbr_3back, np.expand_dims(raw_data_hbr_3back[:,:],axis=0)),axis=0)

# Shape of data (n_subjects, channels, samples)
print("This is the shape of 0back HBO data", data_hbo_0back.shape)
print("This is the shape of 0back HBR data", data_hbr_0back.shape)

print("This is the shape of 1back HBO data", data_hbo_1back.shape)
print("This is the shape of 1back HBR data", data_hbr_1back.shape)

print("This is the shape of 2back HBO data", data_hbo_2back.shape)
print("This is the shape of 2back HBR data", data_hbr_2back.shape)

print("This is the shape of 3back HBO data", data_hbo_3back.shape)
print("This is the shape of 3back HBR data", data_hbr_3back.shape)

print(bad_epochs)

# Save all the data paths

# Save data path
#clean_path = "/Users/adriana/Documents/DTU/thesis/data_processing/Results_fNIRS/"
clean_path = "H:\\Dokumenter\\data_processing\\Results_fNIRS\\"

# 0back HBO & HBR
np.save(clean_path + "hemo_0back", data_hbo_0back) # includes all the events together
np.save(clean_path + "dehemo_0back", data_hbr_0back) # includes all the events together

# 1back HBO & HBR
np.save(clean_path + "hemo_1back", data_hbo_1back) # includes all the events together
np.save(clean_path + "dehemo_1back", data_hbr_1back) # includes all the events together

# 2back HBO & HBR
np.save(clean_path + "hemo_2back", data_hbo_2back) # includes all the events together
np.save(clean_path + "dehemo_2back", data_hbr_2back) # includes all the events together

# 3back HBO & HBR
np.save(clean_path + "hemo_3back", data_hbo_3back) # includes all the events together
np.save(clean_path + "dehemo_3back", data_hbr_3back) # includes all the events together