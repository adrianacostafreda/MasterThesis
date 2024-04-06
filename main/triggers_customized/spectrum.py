import os
import time

import mne
import matplotlib.pyplot as plt
import numpy as np

from scipy.signal import welch
from scipy.integrate import simps


"""Loads cleaned and epoched EEG data from .fif file.
Extracts features and saves them in a .csv file.
"""

def bandpower_from_psd(psd, freqs, bands, relative=False):
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
        If None, window_sec = (1 / min(band)) * 2
    relative : boolean
        If True, return the relative power (= divided by the total power of the signal).
        If False (default), return the absolute power.

    Return
    ------
    bp : float
        Absolute or relative band power.
    """

    # Type checks
    assert isinstance(bands, list), "bands must be a list of tuple(s)"
    assert isinstance(relative, bool), "relative must be a boolean"

    # Safety checks
    freqs = np.asarray(freqs)
    psd = np.asarray(psd)
    assert freqs.ndim == 1, "freqs must be a 1-D array of shape (n_freqs,)"
    assert psd.shape[-1] == freqs.shape[-1], "n_freqs must be last axis of psd"
    
    # Extract frequencies of interest
    all_freqs = np.hstack([[b[0], b[1]] for b in bands])
    fmin, fmax = min(all_freqs), max(all_freqs)
    idx_good_freq = np.logical_and(freqs >= fmin, freqs <= fmax)
    freqs = freqs[idx_good_freq]
    res = freqs[1] - freqs[0]

    # Trim PSD to frequencies of interest
    psd = psd[..., idx_good_freq]
    
    # Calculate total power
    total_power = simps(psd, dx=res, axis=-1)
    total_power = total_power[np.newaxis, ...]

    # Initialize empty array
    bp = np.zeros((len(bands), *psd.shape[:-1]), dtype=np.float64)

    # Enumerate over the frequency bands
    labels = []
    for i, band in enumerate(bands):
        b0, b1, la = band
        labels.append(la)
        idx_band = np.logical_and(freqs >= b0, freqs <= b1)
        bp[i] = simps(psd[..., idx_band], dx=res, axis=-1)

    if relative:
        # If we want to have the bandpower as relative power
        bp /= total_power

    return bp

mne.set_config("MNE_BROWSER_BACKEND", "qt")

path_hc_baseline_eeg="/Users/adriana/Documents/GitHub/MasterThesis/"
#path_hc_baseline_eeg="H:\\Dokumenter\\Visual_Studio\\data_processing_eeg\\.venv\\"
folder_hc_baseline_eeg = os.fsencode(path_hc_baseline_eeg)

# List
psd_epochs = list()
freqs_list = list()

theta_power_rel_list = list()
alpha_power_rel_list = list()

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
        #chan = epochs.info["ch_names"]
        chan = ['AFp1', 'AFp2','AF7','AFF5h','AFF1h', 'AFF2h','AF8','AFF6h','FFC1h','FFC2h',
                'FT7', 'FC5', 'FC3', 'FCC3h',  'FCC1h', 'FCC2h', 
                'FCC4h','FC4','FC6','FT8','CCP3h','CCP1h', 'CCP2h',
                'CCP4h','CP1', 'CP2', 'CPP3h', 'CPP4h', 'P1', 'P2', 'TP7','TP8']

        win = int(4 * sf)  # Window size is set to 4 seconds
        freqs, psd = welch(data, sf, nperseg=win, axis=-1) 

        freqs.shape, psd.shape

        # Let's have a look at the data
        print('Chan = ', chan)
        print('Sampling frequency = ', sf, ' Hz')
        print('Data Shape = ', data.shape)
        print('PSD data Check = ', psd.shape)
        print('Check freqs shape = ', freqs.shape)


        # ---------------Bandpower--------------

        bands=[(0.5, 4, "Delta"), (4, 8, "Theta"), (8, 12, "Alpha"), (12, 16, "Sigma"), (16, 30, "Beta")]

        bp_relative = bandpower_from_psd(psd, freqs, bands, relative=True)

        # Average the psd across all epochs 
        psd_mean = psd.mean(axis=-2)[0, :]
        # Save each psd_mean in the list for each epoch
        psd_epochs.append(psd_mean)

        freqs_list.append(freqs)
        
        # Average delta, theta, alpha, sigma, beta power across all epochs for frontal channel
        #delta_power_rel = bp_relative.mean(axis=-2)[0,:]
        theta_power_rel = bp_relative.mean(axis=-2)[1,0:8] 
        alpha_power_rel = bp_relative.mean(axis=-2)[2,20:26] 
        #sigma_power_rel = bp_relative.mean(axis=-2)[3,:]
        #beta_power_rel = bp_relative.mean(axis=-2)[4,:]

        # Absolute power 
        bp_absolute = bandpower_from_psd(psd, freqs, bands, relative=False)
        
        # Average delta, theta, alpha, sigma, beta power across all epochs for frontal channel
        #delta_power_abs = bp_absolute.mean(axis=-2)[0,0:10]
        theta_power_abs = bp_absolute.mean(axis=-2)[1,0:8] 
        alpha_power_abs = bp_absolute.mean(axis=-2)[2,20:26] 
        #sigma_power_abs = bp_absolute.mean(axis=-2)[3,0:10]
        #beta_power_abs = bp_absolute.mean(axis=-2)[4,0:10]

        # Append relative power to list
        theta_power_rel_list.append(theta_power_rel)
        alpha_power_rel_list.append(alpha_power_rel)

for i in range(len(psd_epochs)):
    plt.plot(freqs_list[0], psd_epochs[i], lw=2)
    legend_labels = [f"Psd for Back Test {i+1}" for i in range(len(psd_epochs))]

plt.xlabel('Frequency (Hz)')
plt.ylabel('Power spectral density (uV^2 / Hz)')
plt.xlim([0, 30])
plt.title("Welch's periodogram")
plt.legend(legend_labels)
plt.show()

time_array = np.linspace(-0.2, 0.8, len(time))

# Plot Theta relative power versus time
for i in range(len(theta_power_rel_list)):
    # Interpolate theta_power_rel to match the length of time_array
    interp_theta_power_rel = np.interp(time_array, np.linspace(-0.2, 0.8, len(theta_power_rel_list[i])), theta_power_rel_list[i])
    plt.plot(time_array, interp_theta_power_rel)
    legend_labels = [f"Theta Relative Power for Back Test {i+1}" for i in range(len(theta_power_rel_list))]

# Plotting
plt.xlabel('Time (s)')
plt.ylabel('Theta Relative Power')
plt.title('Interpolated Theta Relative Power vs Epoched Time')
plt.legend(legend_labels)
plt.grid(True)
plt.show()

# Plot Alpha relative power versus time
for i in range(len(alpha_power_rel_list)):
    # Interpolate theta_power_rel to match the length of time_array
    interp_alpha_power_rel = np.interp(time_array, np.linspace(-0.2, 0.8, len(alpha_power_rel_list[i])), alpha_power_rel_list[i])
    plt.plot(time_array, interp_alpha_power_rel)
    legend_labels = [f"Theta Relative Power for Back Test {i+1}" for i in range(len(alpha_power_rel_list))]

# Plotting
plt.xlabel('Time (s)')
plt.ylabel('Alpha Relative Power')
plt.title('Interpolated Alpha Relative Power vs Epoched Time')
plt.legend(legend_labels)
plt.grid(True)
plt.show()

# Compare Theta and Beta Relative Power
legend_labels = []  # Initialize legend labels list
# Plot Theta relative power versus time
for i in range(len(theta_power_rel_list)):
    # Interpolate theta_power_rel to match the length of time_array
    interp_theta_power_rel = np.interp(time_array, np.linspace(-0.2, 0.8, len(theta_power_rel_list[i])), theta_power_rel_list[i])
    plt.plot(time_array, interp_theta_power_rel, lw=2, color='k')
    legend_labels.append(f"Theta Relative Power for Back Test {i+1}")
# Plot Alpha relative power versus time
for i in range(len(alpha_power_rel_list)):
    # Interpolate theta_power_rel to match the length of time_array
    interp_alpha_power_rel = np.interp(time_array, np.linspace(-0.2, 0.8, len(alpha_power_rel_list[i])), alpha_power_rel_list[i])
    plt.plot(time_array, interp_alpha_power_rel, lw=2, color='r')
    legend_labels.append(f"Alpha Relative Power for Back Test {i+1}")

# Plotting
plt.xlabel('Time (s)')
plt.ylabel('Theta & Alpha Relative Power')
plt.title('Interpolated Theta & Alpha Relative Power vs Epoched Time')
plt.legend(legend_labels)
plt.grid(True)
plt.show()






        



