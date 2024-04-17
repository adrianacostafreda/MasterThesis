import os
import numpy as np
import mne
import matplotlib.pyplot as plt

mne.set_config("MNE_BROWSER_BACKEND", "qt")

#path_hc_baseline_eeg="/Users/adriana/Documents/GitHub/MasterThesis/"
path_hc_baseline_eeg="H:\\Dokumenter\\Visual_Studio\\data_processing_eeg\\.venv\\"
folder_hc_baseline_eeg = os.fsencode(path_hc_baseline_eeg)
filenames = list()

def snr_spectrum(psd, noise_n_neighbor_freqs=1, noise_skip_neighbor_freqs=1):
    """Compute SNR spectrum from PSD spectrum using convolution.

    Parameters
    ----------
    psd : ndarray, shape ([n_trials, n_channels,] n_frequency_bins)
        Data object containing PSD values. Works with arrays as produced by
        MNE's PSD functions or channel/trial subsets.
    noise_n_neighbor_freqs : int
        Number of neighboring frequencies used to compute noise level.
        increment by one to add one frequency bin ON BOTH SIDES
    noise_skip_neighbor_freqs : int
        set this >=1 if you want to exclude the immediately neighboring
        frequency bins in noise level calculation

    Returns
    -------
    snr : ndarray, shape ([n_trials, n_channels,] n_frequency_bins)
        Array containing SNR for all epochs, channels, frequency bins.
        NaN for frequencies on the edges, that do not have enough neighbors on
        one side to calculate SNR.
    """
    # Construct a kernel that calculates the mean of the neighboring
    # frequencies
    averaging_kernel = np.concatenate(
        (
            np.ones(noise_n_neighbor_freqs),
            np.zeros(2 * noise_skip_neighbor_freqs + 1),
            np.ones(noise_n_neighbor_freqs),
        )
    )
    averaging_kernel /= averaging_kernel.sum()

    # Calculate the mean of the neighboring frequencies by convolving with the
    # averaging kernel.
    mean_noise = np.apply_along_axis(
        lambda psd_: np.convolve(psd_, averaging_kernel, mode="valid"), axis=-1, arr=psd
    )

    # The mean is not defined on the edges so we will pad it with nas. The
    # padding needs to be done for the last dimension only so we set it to
    # (0, 0) for the other ones.
    edge_width = noise_n_neighbor_freqs + noise_skip_neighbor_freqs
    pad_width = [(0, 0)] * (mean_noise.ndim - 1) + [(edge_width, edge_width)]
    mean_noise = np.pad(mean_noise, pad_width=pad_width, constant_values=np.nan)

    return psd / mean_noise


for file in os.listdir(folder_hc_baseline_eeg):
    filename = os.fsdecode(file)
    print(filename)
    
    if filename.endswith(".fif"):
        fname = path_hc_baseline_eeg + filename
        print("This is the filename", fname)
        filenames.append(filename)

        # ---- Epoch ----
        epochs = mne.read_epochs(fname)
        #epochs.plot(block=True, events=True)

        # ---- Frequency Analysis ----
        sf = epochs.info["sfreq"]
        time = epochs.times #time in seconds 
        data = epochs.get_data(units = "uV") # convert from V to uV
        event_id = epochs.event_id
        print("This is the event id", event_id)
        print("This is the event id", event_id)
        #chan = epochs.info["ch_names"]
        chan = ['AFp1', 'AFp2','AF7','AFF5h','AFF1h', 'AFF2h','AF8','AFF6h','FFC1h','FFC2h',
                'FT7', 'FC5', 'FC3', 'FCC3h',  'FCC1h', 'FCC2h', 
                'FCC4h','FC4','FC6','FT8','CCP3h','CCP1h', 'CCP2h',
                'CCP4h','CP1', 'CP2', 'CPP3h', 'CPP4h', 'P1', 'P2', 'TP7','TP8']

        tmin = -0.2
        tmax = 0.8
        fmin = 1.0
        fmax = 70.0
        
        epochs_spectrum = epochs.compute_psd(
                                        "welch",
                                        n_fft=int(sf * (tmax - tmin)),
                                        n_overlap=0,
                                        n_per_seg=None,
                                        tmin=tmin,
                                        tmax=tmax,
                                        fmin=fmin,
                                        fmax=fmax,
                                        window="boxcar",
                                        verbose=False,
                                    )
        

        psds, freqs = epochs_spectrum.get_data(return_freqs=True, exclude = [])

        snrs = snr_spectrum(psds, noise_n_neighbor_freqs=3, noise_skip_neighbor_freqs=1)

        fig, axes = plt.subplots(2, 1, sharex="all", sharey="none", figsize=(8, 5))
        
        psds_plot = 10 * np.log10(psds)
        psds_mean = psds_plot.mean(axis=(0, 1))
        psds_std = psds_plot.std(axis=(0, 1))
        axes[0].plot(freqs, psds_mean, color="b")
        axes[0].fill_between(
            freqs, psds_mean - psds_std, psds_mean + psds_std, color="b", alpha=0.2
        )
        axes[0].set(title="PSD spectrum", ylabel="Power Spectral Density [dB]")

        # SNR spectrum
        snr_mean = snrs.mean(axis=(0, 1))
        print("the shape of snr_mean", snr_mean.shape)
        snr_std = snrs.std(axis=(0, 1))

        axes[1].plot(freqs, snr_mean, color="r")
        axes[1].fill_between(
            freqs, snr_mean - snr_std, snr_mean + snr_std, color="r", alpha=0.2
        )
        axes[1].set(
                    title=filename,
                    xlabel="Frequency [Hz]",
                    ylabel="SNR",
                    ylim=[-2, 30],
                    xlim=[fmin, fmax],
        )
        fig.show()
        plt.show()

        stim_freq = 10.2

        # find index of frequency bin closest to stimulation frequency
        i_bin_10hz = np.argmin(abs(freqs - stim_freq))

        i_trial = np.where(epochs.events[:, 2] == event_id["1"])[0]

        pick_channels = mne.pick_types(
                epochs.info, eeg=True, stim=False, exclude="bads", selection=chan
                )

        # get average SNR at 12 Hz for ALL channels
        #snrs_10hz = snrs[i_trial, :, i_bin_10hz]
        #print(snrs_10hz.shape)
        #snrs_10hz_chaverage = snrs_10hz.mean(axis=0)

        #snrs_target = snrs[i_trial, :, i_bin_10hz][:, pick_channels]
    
        # plot SNR topography
        #fig, ax = plt.subplots(1)
        #mne.viz.plot_topomap(snrs_10hz_chaverage, epochs.info, vlim=(1, None), axes=ax)
        #plt.show()

        psds = psds[i_trial,:,i_bin_10hz]
        psds_average = psds.mean(axis=0)

        # plot PSDS topography
        fig, ax = plt.subplots(1)
        mne.viz.plot_topomap(psds_average, epochs.info, axes=ax)
        plt.show()
        







