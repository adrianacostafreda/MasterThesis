import os
import time

import mne
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from scipy import signal
from scipy.signal import welch
from scipy.integrate import simps
import yasa
import seaborn as sns


"""Loads cleaned and epoched EEG data from .fif file.
Extracts features and saves them in a .csv file.
"""

def bandpower(data, sf, window_sec, chan, dB):
    """Compute the average power of the signal x in a specific frequency band.

    Parameters
    ----------
    data : 1d-array
        Input signal in the time-domain.
    sf : float
        Sampling frequency of the data.
    window_sec : float
        Length of each window in seconds.
        
    Return
    ------
    bp : float
        relative band power.
    """

    # Calculate Welch's PSD

    freqs, psd = welch(data, sf, nperseg=window_sec)
    psd=psd[-1]

    print(psd.shape)
    
    print(freqs.shape, psd.shape) # psd has shape (n_channels, n_frequencies)

    # Plot
    plt.plot(freqs, psd[1], 'k', lw=2)
    plt.fill_between(freqs, psd[1], cmap='Spectral')
    plt.xlim(0, 50)
    plt.yscale('log')
    sns.despine()
    plt.title(chan[1])
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('PSD log($uV^2$/Hz)')
    plt.show()

    if dB: 
        psd = 10*np.log10(psd)
        sharey=True

    bp = yasa.bandpower_from_psd(psd, freqs, ch_names=chan)

    return bp


#path_hc_baseline_eeg="/Users/adriana/Documents/GitHub/MasterThesis/"
path_hc_baseline_eeg="H:\\Dokumenter\\Visual_Studio\\data_processing_eeg\\.venv\\"

folder_hc_baseline_eeg = os.fsencode(path_hc_baseline_eeg)

for file in os.listdir(folder_hc_baseline_eeg):
    filename = os.fsdecode(file)
    print(filename)

    if filename.endswith(".fif"):
        fname = path_hc_baseline_eeg + filename

        mne.set_config("MNE_BROWSER_BACKEND", "qt")
    
        feature_list = []
        features = []

        # ---- Epoch ----

        epochs = mne.read_epochs(fname)
        #epochs.plot(block=True, events=True)

        # ---- Split by event type ----

        epochs_0back = epochs['Trigger 4']
        epochs_1back = epochs['Trigger 5']
        epochs_2back = epochs['Trigger 6']
        epochs_3back = epochs['Trigger 7']
        event_count = len(epochs.selection)
        del epochs

        epochs_0back.crop(tmin=0, tmax=40)
        epochs_1back.crop(tmin=0, tmax=55)
        epochs_2back.crop(tmin=0, tmax=55)
        epochs_3back.crop(tmin=0, tmax=55)

        # ---- Frequency Analysis ----

        sfreq = epochs_0back.info["sfreq"]
        time = epochs_0back.times #time in seconds 
        window_sec = 4 * sfreq

        # ---- 0-back Test ----
        data_0back = epochs_0back.get_data(units="uV")
        chan = epochs_0back.info["ch_names"]

        # Relative power
        bp = bandpower(data_0back, sfreq, ch_names = chan, dB=False)
        bp.to_csv("bp_relative.csv")

        # Absolute power 
        bp_absolute = bandpower(data_0back,sfreq, ch_names = chan, relative=False)
        bp_absolute.to_csv("bp_absolute.csv")

        # We can quickly recover the physical bandpower using the `TotalAbsPow`
        bands = ['Delta', 'Theta', 'Alpha', 'Sigma', 'Beta', 'Gamma']
        bp_abs = (bp[bands] * bp['TotalAbsPow'].values[..., None])
        bp_abs.to_csv("total_absolute_power.csv")

        






        



