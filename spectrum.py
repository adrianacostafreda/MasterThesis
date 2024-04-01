import os
import time

import mne
import pandas as pd
import antropy
import matplotlib.pyplot as plt
import numpy as np

from scipy import signal
from scipy.signal import welch
from scipy.integrate import simps
import yasa
import seaborn as sns
sns.set(style='white', font_scale=1.2)


"""Loads cleaned and epoched EEG data from .fif file.
Extracts features and saves them in a .csv file.
"""

def bandpower(data, sf, window_sec,chan):
    """Compute the average power of the signal x in a specific frequency band.

    Parameters
    ----------
    data : 1d-array
        Input signal in the time-domain.
    sf : float
        Sampling frequency of the data.
    band : list
        Lower and upper frequencies of the band of interest.
    window_sec : float
        Length of each window in seconds.
        
    Return
    ------
    bp : float
        relative band power.
    """

    #band = np.asarray(band)
    #low, high = band

    # Calculate Welch's PSD

    # Compute the modified periodogram (Welch)
    freqs, psd = welch(data, sf, nperseg=window_sec, average='median')
    psd=psd[-1]
    
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

    """# Extract bandpower
    psd=psd[-1]
    # Find closest indices of band in frequency vector
    idx_band = np.logical_and(freqs >= low, freqs <= high)

    # Frequency resolution
    freq_res = freqs[1] - freqs[0]

    # Integral approximation of the spectrum using Simpson's rule.
    bp = simps(psd[idx_band], dx=freq_res)

    total_power = simps(psd, dx=freq_res)
    rel_power = bp / total_power

    return rel_power"""
    bp = yasa.bandpower_from_psd(psd, freqs, ch_names=chan)

    return bp


path_hc_baseline_eeg="/Users/adriana/Documents/GitHub/MasterThesis/"
folder_hc_baseline_eeg = os.fsencode(path_hc_baseline_eeg)

for file in os.listdir(folder_hc_baseline_eeg):
    filename = os.fsdecode(file)

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

        data = epochs_0back.get_data(units="uV")
        chan = epochs_0back.info["ch_names"]

        bp = bandpower(data, sfreq,window_sec,chan)
        print(bp)

        # We can quickly recover the physical bandpower using the `TotalAbsPow`
        bands = ['Delta', 'Theta', 'Alpha', 'Sigma', 'Beta', 'Gamma']
        bp_abs = (bp[bands] * bp['TotalAbsPow'].values[..., None])
        print(bp_abs)

        """delta=[0.5,4]
        theta=[4,8]
        alpha=[8,12]
        beta=[12,30]

        #---- 0-Back test ----
        epochs_0back.compute_psd(fmin=2.0, fmax=35.0).plot(average=True, picks="data", exclude="bads")
        #epochs_0back.compute_psd().plot_topomap(ch_type="eeg", normalize=False, contours=0)
        #plt.show()

        spectrum_0back = list()
        
        spectrum_0back.append(bandpower(epochs_0back, sfreq, delta, window_sec))
        spectrum_0back.append(bandpower(epochs_0back, sfreq, theta, window_sec))
        spectrum_0back.append(bandpower(epochs_0back, sfreq, alpha, window_sec))
        spectrum_0back.append(bandpower(epochs_0back, sfreq, beta, window_sec))

        #print('Relative delta power: %.3f, Relative theta power: %.3f, Relative alpha power: %.3f, Relative beta power: %.3f' % (spectrum_0back[0], spectrum_0back[1], spectrum_0back[2], spectrum_0back[3]))

         #---- 1-Back test ----
        epochs_1back.compute_psd(fmin=2.0, fmax=35.0).plot(average=True, picks="data", exclude="bads")
        #epochs_1back.compute_psd().plot_topomap(ch_type="eeg", normalize=False, contours=0)
        #plt.show()

        spectrum_1back = list()
        
        spectrum_1back.append(bandpower(epochs_1back, sfreq, delta, window_sec))
        spectrum_1back.append(bandpower(epochs_1back, sfreq, theta, window_sec))
        spectrum_1back.append(bandpower(epochs_1back, sfreq, alpha, window_sec))
        spectrum_1back.append(bandpower(epochs_1back, sfreq, beta, window_sec))

        #print('Relative delta power: %.3f, Relative theta power: %.3f, Relative alpha power: %.3f, Relative beta power: %.3f' % (spectrum_1back[0], spectrum_1back[1], spectrum_1back[2], spectrum_1back[3]))

         #---- 2-Back test ----
        epochs_2back.compute_psd(fmin=2.0, fmax=35.0).plot(average=True, picks="data", exclude="bads")
        #epochs_2back.compute_psd().plot_topomap(ch_type="eeg", normalize=False, contours=0)
        #plt.show()

        spectrum_2back = list()
        
        spectrum_2back.append(bandpower(epochs_2back, sfreq, delta, window_sec))
        spectrum_2back.append(bandpower(epochs_2back, sfreq, theta, window_sec))
        spectrum_2back.append(bandpower(epochs_2back, sfreq, alpha, window_sec))
        spectrum_2back.append(bandpower(epochs_2back, sfreq, beta, window_sec))

        #print('Relative delta power: %.3f, Relative theta power: %.3f, Relative alpha power: %.3f, Relative beta power: %.3f' % (spectrum_2back[0], spectrum_2back[1], spectrum_2back[2], spectrum_2back[3]))

        #---- 3-Back test ----
        epochs_3back.compute_psd(fmin=2.0, fmax=35.0).plot(average=True, picks="data", exclude="bads")
        #epochs_3back.compute_psd().plot_topomap(ch_type="eeg", normalize=False, contours=0)
        #plt.show()

        spectrum_3back = list()
        
        spectrum_3back.append(bandpower(epochs_3back, sfreq, delta, window_sec))
        spectrum_3back.append(bandpower(epochs_3back, sfreq, theta, window_sec))
        spectrum_3back.append(bandpower(epochs_3back, sfreq, alpha, window_sec))
        spectrum_3back.append(bandpower(epochs_3back, sfreq, beta, window_sec))

        #print('Relative delta power: %.3f, Relative theta power: %.3f, Relative alpha power: %.3f, Relative beta power: %.3f' % (spectrum_3back[0], spectrum_3back[1], spectrum_3back[2], spectrum_3back[3]))
        """






        



