import os
import mne
import matplotlib.pyplot as plt
import numpy as np
import antropy
from scipy.signal import welch
import yasa
import pandas as pd


def compute_brain_wave_power(epochs):
    """
    Goes through all epochs and integrate the PSD using a step function for each channel.
    Returns
    -------
        The total delta, theta, and alpha band powers.
    """
    
    # ---- Frequency Analysis ----
    
    epochs_spectrum = epochs.compute_psd(method = "welch", fmin=1, fmax=30)
    #epochs_spectrum
    #epochs_spectrum.plot_topomap(normalize=True)
    #plt.show()

    mean_spectrum = epochs_spectrum.average()
    psds, freqs = mean_spectrum.get_data(return_freqs=True)
    print(f"\nPSDs shape: {psds.shape}, freqs shape: {freqs.shape}")
        
    # Convert to dB and take mean & standard deviation across channels
    psds = 10*np.log10(psds)
    psds_mean = psds.mean(axis=0)
    psds_mean_list.append(psds_mean)
    freqs_list.append(freqs)
    

    return (psds_mean, freqs)


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

    if dB == (True): 
        psd = 10*np.log10(psd)

    # Extract bandpower: now that we have the PSD for each channel, we need to calculate the average power in the specified bands.
    
    """
    The power in a frequency band is defined by the area under the curve of the non-log-transformed PSD. 
    """

    bands = [(0.5,4,'Delta'), (4,8,'Theta'), (8,12,'Alpha'), (12,16,'Sigma'), (16,30,'Beta') ]
    
    # this function exctracts the average bandpower in specified bands from the pre-computed PSD
    # Calculate the bandpower on 3-D PSD Array

    if relative == (True): 
        bp = yasa.bandpower_from_psd_ndarray(psd, freqs, bands, relative=True) #Relative True -> Relative Power
        np.round(bp,2)
    else:
        bp = yasa.bandpower_from_psd_ndarray(psd, freqs, bands, relative=False) #Relative False -> Absolute Power
        np.round(bp,2)
    
    print("This is the shape of the 3-D bandpower array", bp.shape)

    return (bp, psd, freqs)

def compute_entropy_features(epochs):
    """
    Goes through all epochs and compute the normalized spectral entropy, permutation entropy, and number of zero crsossings.
    (Per channel)
    """
    se = 0
    pe = 0
    zc = 0
    
    for event_id in range(epochs._data.shape[0]):
        for channel_id in range(epochs._data[event_id].shape[0]):
            x = epochs._data[event_id][channel_id]
            se += antropy.spectral_entropy(x, sf = epochs.info['sfreq'], method = 'welch', normalize = True)
            pe += antropy.perm_entropy(x, normalize = True)
            zc += antropy.num_zerocross(x, normalize = True)

    se /= epochs._data[event_id].shape[0]
    pe /= epochs._data[event_id].shape[0]
    zc /= epochs._data[event_id].shape[0]

    return (se, pe, zc)


mne.set_config("MNE_BROWSER_BACKEND", "qt")

#path_hc_baseline_eeg="/Users/adriana/Documents/GitHub/MasterThesis/"
path_hc_baseline_eeg="H:\\Dokumenter\\Visual_Studio\\data_processing_eeg\\.venv\\data_test\\"
folder_hc_baseline_eeg = os.fsencode(path_hc_baseline_eeg)
filenames = list()

psds_mean_list = list()
freqs_list = list()
psd_list_yasa = list()
freqs_list_yasa = list()

# Channels list ["FFC1h", "FFC2h", "FCC1h", "FCC2h", "AFF1h", "AFF2h"]
# delta
delta_power = list()
#theta
theta_power = list()
# alpha
alpha_power = list()

feature_list = []

for file in os.listdir(folder_hc_baseline_eeg):
    filename = os.fsdecode(file)
    
    if filename.endswith(".fif"):
        fname = path_hc_baseline_eeg + filename
        print("This is the filename", fname)
        filenames.append(filename)

        base_name, extension = os.path.splitext(filename)

        # Split the base name using '-' as delimiter
        name_parts = base_name.split('-')
        baseline_part = name_parts[1].split('_')
        
        # Assuming you want to check if name_parts[0][2] is an integer
        if str(name_parts[0][2]).isdigit():
            print("name_parts[0][2] is an integer.", name_parts[0][2])
            patient_id = str(name_parts[0][1]) + str(name_parts[0][2])
            patient_id = int(patient_id)
            print("This is the type of patient_id", type(patient_id))
        else:
            print(name_parts[0][2])
            patient_id = int(name_parts[0][1])

        print("This is the patient id", patient_id)

        if len(baseline_part)>1:
            n_back = int(baseline_part[2][0])
            baseline = 1 #is baseline
            
        else:
            n_back= int(name_parts[1][3])
            baseline = 0 #is not baseline

        print("Number nback:", n_back)   

        features = [] # save the features for each file

        # ---- Epoch ----
        epochs = mne.read_epochs(fname)
        #epochs.plot(block=True, events=True)

        features.append(patient_id)
        features.append(n_back)
        features.append(baseline)
        
        sf = epochs.info["sfreq"]
        time = epochs.times #time in seconds 
        #chan = epochs.info["ch_names"]
        chan = ["FFC1h", "FFC2h", "FCC1h", "FCC2h", "AFF1h", "AFF2h"]
        print(chan)

        data = epochs.get_data(picks = chan, units = "uV") # convert from V to uV
        print('Data Shape = ', data.shape)

        event_count = len(epochs.selection)
        print("This is the event count", event_count)
        
        spectrum_power = compute_brain_wave_power(epochs)

        psds_mean_list.append(spectrum_power[0])
        freqs_list.append(spectrum_power[1])

        # ---------------Bandpower--------------

        window=int(2*sf)

        # Relative power
        # Note that TotalAbsPow contains the total absolute (physical) power, summed across all bands
        bp = bandpower(data, sf, window, chan, False, True)

        # Average delta, theta, alpha, sigma, beta power across all epochs for each of the channels
        # numpy arrays
        
        #Delta Power ["FFC1h", "FFC2h", "FCC1h", "FCC2h", "AFF1h", "AFF2h"]
        
        delta_power = bp[0].sum(axis=-2)[0,:]
        # Compute the average for the length of delta_power
        average_delta_power = delta_power.sum() / len(delta_power)
        features.append(average_delta_power)

        #Theta Power
        theta_power = bp[0].sum(axis=-2)[1,:]
        theta_power /= event_count
        average_theta_power = theta_power.sum() / len(theta_power)
        features.append(average_theta_power)

        #Alpha Power
        alpha_power = bp[0].sum(axis=-2)[2,:] 
        alpha_power /= event_count
        average_alpha_power = alpha_power.sum() / len(alpha_power)
        features.append(average_alpha_power)

        # ---- Entropy and nonlinear features ----

        entropies = compute_entropy_features(epochs)
        
        se = entropies[0]
        se /= event_count
        features.append(se)
        pe = entropies[1]
        pe /= event_count
        features.append(pe)
        zc = entropies[2]
        zc /= event_count
        features.append(zc)
        
        feature_list.append(features)
        print("This is the length of features", len(features))

print("This is feature list", len(feature_list))

# ---- Save to file ----

df = pd.DataFrame(feature_list, columns =['patient_id', 'n-back', 'baseline', 'delta', 'theta', 'alpha', 'se', 'pe', 'zc'])
df.to_csv(os.path.join(path_hc_baseline_eeg, "eeg_features.csv"), index = False)

_, ax = plt.subplots()

for i in range(len(psds_mean_list)):
    ax.plot(freqs_list[0], psds_mean_list[i], lw=2)

ax.set(
    title="Welch PSD",
    xlabel="Frequency (Hz)",
    ylabel="Power Spectral Density (dB)",
)
plt.show()
