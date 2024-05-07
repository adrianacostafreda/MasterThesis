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

#path = "/Users/adriana/Documents/DTU/thesis/data_acquisition/fNIRS/healthy_controls/"
path = "H:\\Dokumenter\\data_acquisition\\data_fnirs\\healthy_controls\\baseline\\snirf_files\\"

datapath = DataPath(path, recursive=False)
print(len(datapath.getDataPaths()))

MAX_LENGTH_HEALTHY = 2326

# Loop over all files preprocess them with HemoData and save them at an numpy array.
# It save the data for each patient, for each event, for each channel:  patient X events X channels X time_samples.
for id,file in enumerate(datapath.getDataPaths()):
        raw_haemo = HemoData(file, preprocessing=True, isPloting=False).getMneIoRaw()
        #raw_haemo.plot(block=True)

        # Add trigger 4 for the 0-Back Task
        annot = raw_haemo.annotations
        onset_trigg4 = annot.onset[8] - 15 - 40
        annotations = mne.Annotations(onset=[onset_trigg4 + 15, annot.onset[8] + 15, annot.onset[9] + 15, annot.onset[10] + 15], duration=[20, 20, 20, 20], description=["0", "1", "2", "3"])
        raw_haemo.set_annotations(annotations)

        events, event_dict = mne.events_from_annotations(raw_haemo)
        reverse_event_dict = invertDic(event_dict)

        raw_data_hbo = raw_haemo.get_data(picks=["hbo"]) # channels x samples
        raw_data_hbr = raw_haemo.get_data(picks=["hbr"]) # channels x samples

        # Standardize
        ss = StandardScaler()
        raw_data_hbo = ss.fit_transform(raw_data_hbo.T).T
        # The shape of raw data is (n_channels, samples)
        raw_data_hbr = ss.fit_transform(raw_data_hbr.T).T

        # We will expand each task 15 s which corresponds to 153
        t_interval = 153

        # Simple version subjects x events x channels x samples.
        if id == 0:
            additional_data_hbo = np.expand_dims(raw_data_hbo[:,events[0][0]:events[-1][0]+2*t_interval][:,:MAX_LENGTH_HEALTHY],axis=0)
            additional_data_hbr = np.expand_dims(raw_data_hbr[:,events[0][0]:events[-1][0]+2*t_interval][:,:MAX_LENGTH_HEALTHY],axis=0)
        else:
            additional_data_hbo = np.concatenate((additional_data_hbo,np.expand_dims(raw_data_hbo[:,events[0][0]:events[-1][0]+2*t_interval][:,:MAX_LENGTH_HEALTHY],axis=0)),axis=0)
            additional_data_hbr = np.concatenate((additional_data_hbr,np.expand_dims(raw_data_hbr[:,events[0][0]:events[-1][0]+2*t_interval][:,:MAX_LENGTH_HEALTHY],axis=0)),axis=0)


        # Take each event separetly for each experiment (healthy control/patient). Ends up experiments X events X channels X samples.
        # Look over all events (e.g. 0-Back, 1-Back, 2-Back, 3-Back)

        for idx, event in enumerate(events):
                if idx == 0:
                        event_data_hbo = np.expand_dims(raw_data_hbo[:, event[0]:int(event[0]+t_interval)], axis=0)
                        event_order_hbo = np.expand_dims(reverse_event_dict[event[2]], axis=0)

                        event_data_hbr = np.expand_dims(raw_data_hbr[:, event[0]:int(event[0]+t_interval)], axis=0)
                        event_order_hbr = np.expand_dims(reverse_event_dict[event[2]], axis=0)
                else:
                        event_data_hbo = np.concatenate((event_data_hbo, np.expand_dims(raw_data_hbo[:,event[0]:int(event[0]+t_interval)], axis=0)),axis=0)
                        event_order_hbo = np.concatenate((event_order_hbo, np.expand_dims(reverse_event_dict[event[2]], axis=0)), axis=0)

                        event_data_hbr = np.concatenate((event_data_hbr, np.expand_dims(raw_data_hbr[:,event[0]:int(event[0]+t_interval)], axis=0)),axis=0)
                        event_order_hbr = np.concatenate((event_order_hbr, np.expand_dims(reverse_event_dict[event[2]], axis=0)), axis=0)
                
                
        if id == 0:
                data_hbo = np.expand_dims(event_data_hbo,axis=0)
                event_order_data_hbo = np.expand_dims(event_order_hbo,axis=0)

                data_hbr = np.expand_dims(event_data_hbr,axis=0)
                event_order_data_hbr = np.expand_dims(event_order_hbr,axis=0)
        else:
                data_hbo = np.concatenate((data_hbo, np.expand_dims(event_data_hbo,axis=0)), axis=0)
                event_order_data_hbo = np.concatenate((event_order_data_hbo, np.expand_dims(event_order_hbo,axis=0)), axis=0)

                data_hbr = np.concatenate((data_hbr, np.expand_dims(event_data_hbr, axis=0)), axis=0)
                event_order_data_hbr = np.concatenate((event_order_data_hbr, np.expand_dims(event_order_hbr,axis=0)), axis=0)

# The shape of data is (n_subjects, events, channels, samples)
print("This is the shape of data", data_hbo.shape)
print("This is the shape of data", data_hbr.shape)

# Shape of event order would be (n_subjects, n_events)
print("This is the shape of the event order", event_order_data_hbo.shape)
print("This is the shape of the event order", event_order_data_hbr.shape)

# Shape of additional data (n_subjects, channels, samples)
print("This is the shape of additional data", additional_data_hbo.shape)
print("This is the shape of additional data", additional_data_hbr.shape)

# Save all the data paths

# Save data path
#clean_path = "/Users/adriana/Documents/DTU/thesis/data_processing/Results_fNIRS/"
clean_path = "H:\\Dokumenter\\data_processing\\Results_fNIRS\\"

# HBO
np.save(clean_path + "hemo_per_event", data_hbo) # separates the data for each of the events 
np.save(clean_path + "event_order_hbo", event_order_data_hbo)
np.save(clean_path + "hemo", additional_data_hbo) # includes all the events together

#HBR
np.save(clean_path + "dehemo_per_event", data_hbr) # separates the data for each of the events 
np.save(clean_path + "event_order_hbr", event_order_data_hbr)
np.save(clean_path + "dehemo", additional_data_hbr) # includes all the events together
