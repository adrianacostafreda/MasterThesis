import numpy as np
import mne

#from DataPath import DataPath
#from Hemo import HemoData
from sklearn.preprocessing import StandardScaler

import mne_nirs
from itertools import compress
import matplotlib.pyplot as plt
from mne.decoding import UnsupervisedSpatialFilter
from scipy.spatial.distance import pdist, squareform
from mne._fiff.pick import pick_channels, pick_info, pick_types
from sklearn.decomposition import PCA

import os
from queue import LifoQueue

class DataPath:
    
    def __init__(self, baseline_path: str, fif: bool=False,  recursive: bool=True) -> None:
        self.stack = LifoQueue(maxsize=100)
        self.data_path = list()
        self.baseline_path = baseline_path
        self.stack.put(self.baseline_path)
        self.iter = 0
        self.isFif = fif
        if recursive:
            self.recurrentDirSearch()
        else:
            self.getAllinOneDir()
    
    def getAllinOneDir(self):
        onlyfiles = self.get_immediate_files(self.baseline_path)
        for file in onlyfiles:
            if file.find(".snirf") != -1 or file.find(".nirf") != -1: 
                self.data_path.append(os.path.join(self.baseline_path,file))

    def get_immediate_subdirectories(self, a_dir):
        return [os.path.join(a_dir, name) for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name))]
    
    def get_immediate_files(self, a_dir):
        return [f for f in os.listdir(a_dir) if os.path.isfile(os.path.join(a_dir, f))]

    def isThisTheFinalDir(self, a_dir):
        onlyfiles = self.get_immediate_files(a_dir)
        for file in onlyfiles:
            if not self.isFif:
                if file.find(".snirf") != -1 or file.find(".nirf") != -1:
                    return os.path.join(a_dir,file)
            elif self.isFif:
                if file.find(".fif") != -1:
                    return os.path.join(a_dir,file)
        return None
    
    def recurrentDirSearch(self):
        self.iter += 1 
        if self.stack.empty():
            return self.data_path
        else:
            a_dir = self.stack.get()
            file = self.isThisTheFinalDir(a_dir)
            if file is not None:
                self.data_path.append(file)
            else:
                subDirs = self.get_immediate_subdirectories(a_dir)
                if subDirs is not None:
                    for dir in subDirs:
                        self.stack.put(dir)
            return self.recurrentDirSearch()
        
    def getDataPaths(self):
        return self.data_path
class HemoData():
    """
    It reads an fnirs file and stores the hemoglobin signal. 

    Attributes:
        data (mne.io.Raw): The mne.io.Raw object with the hemoglobin data.
        shortChannels (list): A list with the ID (initial raw data) of the short channels. 
        isPloting (bool): True if you want to plot graphs during initialization.
        useShortChannelRegression (bool): True if you want to use short channel regression during preprocessing.
    """
    
    def __init__(self, path: str, preprocessing: bool=True, isPloting: bool=False, useShortChannelRegression: bool=True) -> None:
        """
        Initializes a HemoData object.
 
        Parameters:
            path (str): The absolute path of the fnirs file.
            preprocessing (bool): True if data are in raw format, False if they are already hemoglobin. 
            isPloting (bool): True if you want to plot graphs during initialization.
            useShortChannelRegression (bool): True if you want to use short channel regression during preprocessing.
        """
        if path == "" or path is None:
            return None
        raw = self.loadData(path)
        self.ploting = isPloting
        self.useShortChannelRegression = useShortChannelRegression
        if preprocessing:
            self.data = self.preprocess(raw)
        else:
            self.data, self.shortChannels = self.removeShortChannels(raw)
            #self.data = raw

    def loadData(self, path: str):
        """
        Loads the snirf file to mne.io.Raw data
 
        Parameters:
            path (str): The absolute path of the fnirs file.
 
        Returns:
            mne.io.Raw: The object with the data.
        """
        raw = mne.io.read_raw_snirf(path, optode_frame="mri")
        raw.load_data()
        return raw

    def preprocess(self, data: mne.io.Raw):
        """
        Aply the preprocessing routine to the data with the raw signals. The routine includes:
            1.  Convert raw to optical density.
            2.  Reject and interpolate bad channels.
            3.  Short channels regression (optional).
            4.  TDDR for the motion artifacts.
            5.  Convert optical density to hemoglobin (ppf=0.15).
            6.  2nd order butterworth bandpass filter [0.01, 0.4]
 
        Parameters:
            data (mne.io.Raw): The mne.io.Raw object with the raw signals.
 
        Returns:
            The mne.io.Raw object with the hemoglobin data.
        """
        # Converting from raw intensity to optical density
        data_od = mne.preprocessing.nirs.optical_density(data)

        # Reject bad channels and interpolate them
        sci = mne.preprocessing.nirs.scalp_coupling_index(data_od)
        data_od.info["bads"] = list(compress(data_od.ch_names, sci < 0.75))
        print(data_od.info)
        print("Bad channels: " + str(data_od.info["bads"]))
        if self.ploting:
            fig, ax = plt.subplots(layout="constrained")
            ax.hist(sci)
            ax.set(xlabel="Scalp Coupling Index", ylabel="Count", xlim=[0, 1])
            plt.show()
        
        # Interpolate
        data_od.interpolate_bads()

        # Short Channel Regression
        if self.useShortChannelRegression:
            print("Perforning short channel regression...")
            data_od = mne_nirs.signal_enhancement.short_channel_regression(data_od)
            print("Done!")

        # Remove short channels 
        data_od, self.shortChannels = self.removeShortChannels(data_od)
        print(data_od.info)

        # Remove motion artifacts
        data_od_corrected = mne.preprocessing.nirs.temporal_derivative_distribution_repair(data_od)

        # Convert optical density to hemoglobin
        data_heamo = mne.preprocessing.nirs.beer_lambert_law(data_od_corrected, ppf=0.05)

        # Physiological noise removal - Bandpass filtering
        iir_params = dict(order=4, ftype='butter')
        data_heamo.filter(l_freq=0.01, h_freq=None, method='iir',
            iir_params=iir_params, verbose=True)
        data_heamo.filter(l_freq=None, h_freq=0.4, method='iir',
            iir_params=iir_params, verbose=True)
        return data_heamo

    def removeShortChannels(self, data: mne.io.Raw):
        """
        Remove the short channels from <data>.
 
        Parameters:
            data (mne.io.Raw): A mne.io.Raw object. 
 
        Returns:
            mne.io.Raw: The mne.io.Raw object without the short channels.
        """
        picks = mne.pick_types(data.info, meg=False, fnirs=True)
        dists = mne.preprocessing.nirs.source_detector_distances(
        data.info, picks=picks
        )
        data.pick(picks[dists > 0.01])
        return data, picks[dists <= 0.01]
    
    def plot(self, show: bool=True, title: str="Hemoglobin Concentration."):
        """
        Plot the signal/time series of each channel of the data.
        """
        self.data.plot(n_channels=len(self.data.ch_names), duration=1000, show_scrollbars=False, title=title, scalings="auto")
        if show:
            plt.show()

    def getShortChannels(self):
        """
        Get the IDs of the short channels.
        """
        return self.shortChannels

    def getMneIoRaw(self):
        """
        Get the mne.io.Raw object with the hemoglobin data.
        """
        return self.data

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
            begin_trigger = annot.onset[idx] + 11 # segment the data 11 seconds after the trigger
            print(annot.duration[idx])
            duration_trigger = annot.duration[idx] + 40 # Durations of the test are 50 seconds
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
    trigger_data["4"]["duration"].append(38) # Duration of 0-Back Test is 48 s

    end_time = trigger_data["4"]["begin"][0] + 38
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
    raw_new.plot(block=True)
    plt.show()
    
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

path = "/Users/adriana/Documents/DTU/thesis/data_acquisition/data_fnirs/healthy_controls/"
#path = "H:\\Dokumenter\\data_acquisition\\data_fnirs\\healthy_controls\\baseline\\snirf_files\\"

datapath = DataPath(path, recursive=False)
print(len(datapath.getDataPaths()))

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

        raw_data_hbo_0back = epochs[0].get_data(picks=["hbo"]) # epochs x channels x samples
        raw_data_hbr_0back = epochs[0].get_data(picks=["hbr"]) # epochs xchannels x samples

        raw_data_hbo_1back = epochs[1].get_data(picks=["hbo"]) # epochs x channels x samples
        raw_data_hbr_1back = epochs[1].get_data(picks=["hbr"]) # epochs x channels x samples

        raw_data_hbo_2back = epochs[2].get_data(picks=["hbo"]) # epochs x channels x samples
        raw_data_hbr_2back = epochs[2].get_data(picks=["hbr"]) # epochs x channels x samples

        raw_data_hbo_3back = epochs[3].get_data(picks=["hbo"]) # epochs x channels x samples
        raw_data_hbr_3back = epochs[3].get_data(picks=["hbr"]) # epochs x channels x samples

        print("This is the shape of raw_data_hbo_0back", raw_data_hbr_0back.shape)

        """# Standardize
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
        raw_data_hbr_3back = ss.fit_transform(raw_data_hbr_3back.T).T"""

        # We will expand each task 15 s which corresponds to 153
        t_interval = 153

        # Simple version subjects x events x channels x samples.
        if id == 0:
            data_hbo_0back = np.expand_dims(raw_data_hbo_0back[:, :, :],axis=0)
            data_hbr_0back = np.expand_dims(raw_data_hbr_0back[:, :, :],axis=0)
        else:
            data_hbo_0back = np.concatenate((data_hbo_0back, np.expand_dims(raw_data_hbo_0back[:, :, :],axis=0)),axis=0)
            data_hbr_0back = np.concatenate((data_hbr_0back, np.expand_dims(raw_data_hbr_0back[:, :, :],axis=0)),axis=0)

        # Simple version subjects x events x channels x samples.
        if id == 0:
            data_hbo_1back = np.expand_dims(raw_data_hbo_1back[:, :, :],axis=0)
            data_hbr_1back = np.expand_dims(raw_data_hbr_1back[:, :, :],axis=0)
        else:
            data_hbo_1back = np.concatenate((data_hbo_1back, np.expand_dims(raw_data_hbo_1back[:, :, :],axis=0)),axis=0)
            data_hbr_1back = np.concatenate((data_hbr_1back, np.expand_dims(raw_data_hbr_1back[:, :, :],axis=0)),axis=0)

        # Simple version subjects x events x channels x samples.
        if id == 0:
            data_hbo_2back = np.expand_dims(raw_data_hbo_2back[:, :, :],axis=0)
            data_hbr_2back = np.expand_dims(raw_data_hbr_2back[:, :, :],axis=0)
        else:
            data_hbo_2back = np.concatenate((data_hbo_2back, np.expand_dims(raw_data_hbo_2back[:, :, :],axis=0)),axis=0)
            data_hbr_2back = np.concatenate((data_hbr_2back, np.expand_dims(raw_data_hbr_2back[:, :, :],axis=0)),axis=0)

        # Simple version subjects x events x channels x samples.
        if id == 0:
            data_hbo_3back = np.expand_dims(raw_data_hbo_3back[:, :, :],axis=0)
            data_hbr_3back = np.expand_dims(raw_data_hbr_3back[:, :, :],axis=0)
        else:
            data_hbo_3back = np.concatenate((data_hbo_3back, np.expand_dims(raw_data_hbo_3back[:, :, :],axis=0)),axis=0)
            data_hbr_3back = np.concatenate((data_hbr_3back, np.expand_dims(raw_data_hbr_3back[:, :, :],axis=0)),axis=0)

# Shape of data (n_subjects, channels, samples)
print("This is the shape of 0back HBO data", data_hbo_0back.shape)
print("This is the shape of 0back HBR data", data_hbr_0back.shape)

print("This is the shape of 1back HBO data", data_hbo_1back.shape)
print("This is the shape of 1back HBR data", data_hbr_1back.shape)

print("This is the shape of 2back HBO data", data_hbo_2back.shape)
print("This is the shape of 2back HBR data", data_hbr_2back.shape)

print("This is the shape of 3back HBO data", data_hbo_3back.shape)
print("This is the shape of 3back HBR data", data_hbr_3back.shape)

# Save all the data paths

# Save data path
clean_path_0back = "/Users/adriana/Documents/DTU/thesis/data_processing/Results_fNIRS/0_back/"
clean_path_1back = "/Users/adriana/Documents/DTU/thesis/data_processing/Results_fNIRS/1_back/"
clean_path_2back = "/Users/adriana/Documents/DTU/thesis/data_processing/Results_fNIRS/2_back/"
clean_path_3back = "/Users/adriana/Documents/DTU/thesis/data_processing/Results_fNIRS/3_back/"
#clean_path = "H:\\Dokumenter\\data_processing\\Results_fNIRS\\"

# 0back HBO & HBR
np.save(clean_path_0back + "hemo_0back", data_hbo_0back) 
np.save(clean_path_0back + "dehemo_0back", data_hbr_0back)

# 1back HBO & HBR
np.save(clean_path_1back + "hemo_1back", data_hbo_1back) 
np.save(clean_path_1back + "dehemo_1back", data_hbr_1back) 

# 2back HBO & HBR
np.save(clean_path_2back + "hemo_2back", data_hbo_2back) 
np.save(clean_path_2back + "dehemo_2back", data_hbr_2back) 

# 3back HBO & HBR
np.save(clean_path_3back + "hemo_3back", data_hbo_3back)
np.save(clean_path_3back + "dehemo_3back", data_hbr_3back) 