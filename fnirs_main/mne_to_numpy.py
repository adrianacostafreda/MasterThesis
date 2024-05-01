import numpy as np
import mne
from DataPath import DataPath
from Hemo import HemoData
from sklearn.preprocessing import StandardScaler

def invertDic(dic: dict):
        '''
        Helping function to invert the keys and the values of a dictionary.
        '''
        inv_dic = {v: k for k, v in dic.items()}
        return inv_dic

path = "/Users/adriana/Documents/DTU/thesis/data_acquisition/fNIRS/healthy_controls/"
datapath = DataPath(path, recursive=False)
print(len(datapath.getDataPaths()))

MAX_LENGTH_HEALTHY = 2326

# Loop over all files preprocess them with HemoData and save them at an numpy array.
# It save the data for each patient, for each event, for each channel:  patient X events X channels X time_samples.
for id,file in enumerate(datapath.getDataPaths()):
        raw_haemo = HemoData(file, preprocessing=True, isPloting=False).getMneIoRaw()
        raw_data = raw_haemo.get_data(picks=["hbo"]) # channels x samples

        annot = raw_haemo.annotations
    
        onset_trigg4 = annot.onset[8] - 15 - 40

        annotations = mne.Annotations(onset=[onset_trigg4, annot.onset[8], annot.onset[9], annot.onset[10]], duration=[48,60,60,60], description=["0", "1", "2", "3"])
        raw_haemo.set_annotations(annotations)

        print("These are new annotations", raw_haemo.annotations)
        raw_haemo.plot(block=True)

        events, event_dict = mne.events_from_annotations(raw_haemo)
        reverse_event_dict = invertDic(event_dict)

        # Standardize
        ss = StandardScaler()
        raw_data = ss.fit_transform(raw_data.T).T
        print(raw_data.shape)
        # The shape of raw data is (n_channels, samples)

        # We will expand each task 15 s which corresponds to 153
        t_interval = 153

        # Simple version subjects x events x channels x samples.
        if id == 0:
            additional_data = np.expand_dims(raw_data[:,events[0][0]:events[-1][0]+2*t_interval][:,:MAX_LENGTH_HEALTHY],axis=0)
        else:
            additional_data = np.concatenate((additional_data,np.expand_dims(raw_data[:,events[0][0]:events[-1][0]+2*t_interval][:,:MAX_LENGTH_HEALTHY],axis=0)),axis=0)

        # Take each event separetly for each experiment (healthy control/patient). Ends up experiments X events X channels X samples.
        # Look over all events (e.g. 0-Back, 1-Back, 2-Back, 3-Back)

        for idx, event in enumerate(events):
                if idx == 0:
                        event_data = np.expand_dims(raw_data[:, event[0]:int(event[0]+t_interval)], axis=0)
                        event_order = np.expand_dims(reverse_event_dict[event[2]], axis=0)
                else:
                        event_data = np.concatenate((event_data, np.expand_dims(raw_data[:,event[0]:int(event[0]+t_interval)], axis=0)),axis=0)
                        event_order = np.concatenate((event_order, np.expand_dims(reverse_event_dict[event[2]], axis=0)), axis=0)
                
                
        if id == 0:
                data = np.expand_dims(event_data,axis=0)
                event_order_data = np.expand_dims(event_order,axis=0)
        else:
                data = np.concatenate((data,np.expand_dims(event_data,axis=0)),axis=0)
                event_order_data = np.concatenate((event_order_data,np.expand_dims(event_order,axis=0)),axis=0)

# The shape of data is (n_subjects, events, channels, samples)
print("This is the shape of data", data.shape)

# Shape of event order would be (n_subjects, n_events)
print("This is the shape of the event order", event_order_data.shape)

# Shape of additional data (n_subjects, channels, samples)
print("This is the shape of additional data", additional_data.shape)

# Save all the data paths

# Save data path
clean_path = "/Users/adriana/Documents/DTU/thesis/data_processing/Results_fNIRS/"

np.save(path + "hemo_per_event", data) # separates the data for each of the events 
np.save(path + "event_order", event_order_data)
np.save(path + "hemo", additional_data) # includes all the events together
