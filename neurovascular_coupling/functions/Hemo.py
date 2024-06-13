import numpy as np
import mne
import mne_nirs
from itertools import compress
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from mne.decoding import UnsupervisedSpatialFilter
from scipy.spatial.distance import pdist, squareform
from mne._fiff.pick import pick_channels, pick_info, pick_types
from sklearn.decomposition import PCA


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
        Apply the preprocessing routine to the data with the raw signals. The routine includes:
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
    

# hemo = HemoData("L:\\LovbeskyttetMapper\\CONNECT-ME\\DTU\\Alex_Data\\HealthyPatients\\data_initial\\C19_raw.snirf", preprocessing=True)
# hemo.plot()
    
# Satori file
# hemo = HemoData("L:\\LovbeskyttetMapper\\CONNECT-ME\\DTU\\Alex_Data\\Satori\\HealthyPatients\\C8_raw_Satori_od_regression.snirf", preprocessing=False)
# hemo.plot()