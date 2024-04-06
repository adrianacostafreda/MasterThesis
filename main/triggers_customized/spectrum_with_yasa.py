import os
import time

import mne
import matplotlib.pyplot as plt
import numpy as np

from scipy.signal import welch
import yasa


"""Loads cleaned and epoched EEG data from .fif file.
Extracts features and saves them in a .csv file.
"""

def bandpower(data, sf, window, chan, dB, relative):
    """Compute the average power of the signal x in a specific frequency band.

    Parameters
    ----------
    data : 1d-array
        Input signal in the time-domain.
    sf : float
        Sampling frequency of the data.
    window_sec : float
        Length of each window in seconds.
    chan: list
        Names of the channels
    dB: bool
        If True, it returns the bandpower in dB
    relative: bool
        If True, it returns the relative band power
        
    Return
    ------
    bp : float
        relative or absolute band power.
    """

    # Calculate Welch's PSD
    freqs, psd = welch(data, sf, nperseg=window, axis=-1)
    # psd unit is uV^2/Hz

    print("This is the shape for the frequencies vector and the psd vector", freqs.shape, psd.shape) 
    # psd has shape (n_channels, n_frequencies)

    if dB == True: 
        psd = 10*np.log10(psd)

    # Extract bandpower: now that we have the PSD for each channel, we need to calculate the average power in the specified bands.
    
    """
    The power in a frequency band is defined by the area under the curve of the non-log-transformed PSD. 
    """

    bands = [(0.5,4,'Delta'), (4,8,'Theta'), (8,12,'Alpha'), (12,16,'Sigma'), (16,30,'Beta') ]
    
    # this function exctracts the average bandpower in specified bands from the pre-computed PSD
    # Calculate the bandpower on 3-D PSD Array

    if relative == True: 
        bp = yasa.bandpower_from_psd_ndarray(psd, freqs, bands, relative=True) #Relative True -> Relative Power
        np.round(bp,2)
    else:
        bp = yasa.bandpower_from_psd_ndarray(psd, freqs, bands, relative=False) #Relative True -> Relative Power
        np.round(bp,2)


    print("This is the shape of the 3-D bandpower array", bp.shape)

    return (bp, psd, freqs)


mne.set_config("MNE_BROWSER_BACKEND", "qt")

path_hc_baseline_eeg="/Users/adriana/Documents/GitHub/MasterThesis/"
#path_hc_baseline_eeg="H:\\Dokumenter\\Visual_Studio\\data_processing_eeg\\.venv\\"

folder_hc_baseline_eeg = os.fsencode(path_hc_baseline_eeg)

psd_list = list()
freqs_list = list()
theta_power_abs_epochs=()

count = 0

for file in os.listdir(folder_hc_baseline_eeg):
    filename = os.fsdecode(file)
    
    if filename.endswith(".fif"):
        fname = path_hc_baseline_eeg + filename
        print("This is the filename", fname)

        
        # ---- Epoch ----
        epochs = mne.read_epochs(fname)
        #epochs.plot(block=True, events=True)

        # ---- Frequency Analysis ----
        sf = epochs.info["sfreq"]
        time = epochs.times #time in seconds 
        data = epochs.get_data(units = "uV") # convert from V to uV
        chan = epochs.info["ch_names"]

        # Let's have a look at the data
        print('Chan = ', chan)
        print('Sampling frequency = ', sf, ' Hz')
        print('Data Shape = ', data.shape)

        # ---------------Bandpower--------------

        window=int(2*sf)

        # Relative power
        # Note that TotalAbsPow contains the total absolute (physical) power, summed across all bands
        bp_relative = bandpower(data, sf, window, chan, False, True)

        # Average delta, theta, alpha, sigma, beta power across all epochs for each of the channels
        # numpy arrays
        delta_power_rel = bp_relative[0].mean(axis=-2)[0,:]
        theta_power_rel = bp_relative[0].mean(axis=-2)[1,:] 
        alpha_power_rel = bp_relative[0].mean(axis=-2)[2,:] 
        sigma_power_rel = bp_relative[0].mean(axis=-2)[3,:]
        beta_power_rel = bp_relative[0].mean(axis=-2)[4,:]
        
        # Absolute power 
        bp_absolute = bandpower(data, sf, window, chan, False, False)
        psd = bp_absolute[1]
        freqs = bp_absolute[2]
        
        # Make the mean of psd across all epochs
        psd_power = psd.mean(axis=-2)[-1]
            
        psd_list.append(psd_power)
        freqs_list.append(freqs)
        
        # Average delta, theta, alpha, sigma, beta power across all epochs for each of the channels
        # numpy arrays
        delta_power_abs = bp_absolute[0].mean(axis=-2)[0,:]
        theta_power_abs = bp_absolute[0].mean(axis=-2)[1,:] 
        alpha_power_abs = bp_absolute[0].mean(axis=-2)[2,:] 
        sigma_power_abs = bp_absolute[0].mean(axis=-2)[3,:]
        beta_power_abs = bp_absolute[0].mean(axis=-2)[4,:]

        """# ----------------------------------------------------dB------------------------------------------------------

        # Relative power
        # Note that TotalAbsPow contains the total absolute (physical) power, summed across all bands
        bp_relative_dB = bandpower(data, sf, window, chan, True, True)

        # Average delta, theta, alpha, sigma, beta power across all epochs for each of the channels
        # numpy arrays
        delta_power_rel_dB = bp_relative_dB.mean(axis=-2)[0,:]
        theta_power_rel_dB = bp_relative_dB.mean(axis=-2)[1,:] 
        alpha_power_rel_dB = bp_relative_dB.mean(axis=-2)[2,:] 
        sigma_power_rel_dB = bp_relative_dB.mean(axis=-2)[3,:]
        beta_power_rel_dB = bp_relative_dB.mean(axis=-2)[4,:]
        
        # Absolute power 
        bp_absolute_dB = bandpower(data, sf, window, chan, True, False)
        
        # Average delta, theta, alpha, sigma, beta power across all epochs for each of the channels
        # numpy arrays
        delta_power_abs_dB = bp_absolute_dB.mean(axis=-2)[0,:]
        theta_power_abs_dB = bp_absolute_dB.mean(axis=-2)[1,:] 
        alpha_power_abs_dB = bp_absolute_dB.mean(axis=-2)[2,:] 
        sigma_power_abs_dB = bp_absolute_dB.mean(axis=-2)[3,:]
        beta_power_abs_dB = bp_absolute_dB.mean(axis=-2)[4,:]"""

for i in range(len(psd_list)):
    plt.plot(freqs_list[0], psd_list[i])
    legend_labels = [f"PSD Back Test {i+1}" for i in range(len(psd_list))]

plt.xlabel("Frequency (Hz)")
plt.ylabel("Power uV^2/Hz")
plt.xlim(0, 30)
plt.title("PSD Welch's Method")
plt.legend(legend_labels)
plt.show()










        



