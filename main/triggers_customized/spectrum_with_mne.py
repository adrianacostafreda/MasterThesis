import os
import time

import mne
import matplotlib.pyplot as plt
import numpy as np

mne.set_config("MNE_BROWSER_BACKEND", "qt")

path_hc_baseline_eeg="/Users/adriana/Documents/GitHub/MasterThesis/"
#path_hc_baseline_eeg="H:\\Dokumenter\\Visual_Studio\\data_processing_eeg\\.venv\\"
folder_hc_baseline_eeg = os.fsencode(path_hc_baseline_eeg)

psds_mean_list=list()
freqs_list = list()

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

        
        epochs_spectrum = epochs.compute_psd(method = "welch", fmin=1, fmax=50)
        epochs_spectrum
        epochs_spectrum.plot_topomap(normalize=True)
        plt.show()

        mean_spectrum = epochs_spectrum.average()
        psds, freqs = mean_spectrum.get_data(return_freqs=True)
        print(f"\nPSDs shape: {psds.shape}, freqs shape: {freqs.shape}")
        
        # Convert to dB and take mean & standard deviation across channels
        psds = 10*np.log10(psds)
        psds_mean = psds.mean(axis=0)
        psds_mean_list.append(psds_mean)
        freqs_list.append(freqs)


_, ax = plt.subplots()

for i in range(len(psds_mean_list)):
    ax.plot(freqs_list[0], psds_mean_list[i], lw=2)
    legend_labels= [f"Psds dB for Back Test {i+1}" for i in range(len(psds_mean_list))]

ax.set(
    title="Welch PSD (eeg)",
    xlabel="Frequency (Hz)",
    ylabel="Power Spectral Density (dB)",
)
ax.legend(legend_labels)
plt.show()

