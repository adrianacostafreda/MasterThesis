import mne
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

def calculate_psd(epochs, subjectname, fminmax=[1,50], method="welch",window="hamming", 
                  window_duration=2, window_overlap=0.5,zero_padding=3, tminmax=[None,None],
                  verbose = False, plot = True):
    """
    Calculate power spectrum density with FFT/Welch's method and plot the result.

    Parameters
    ----------
    epochs: Epochs type (MNE-Python) EEG file
    fminfmax: The minimum and maximum frequency range for estimating Welch's PSD
    window: The window type for estimating Welch's PSD
    window_duration: An integer for the length of the window
    window_overlap: A float for the percentage of window size for overlap between te windows 
    zero-padding: A float for coefficient times window size for zero-pads
    tminmax : A list of first and last timepoint of the epoch to include; uses all epochs by default


    Returns
    -------
    psds: An array for power spectrum density values 
    freqs: An array for the corresponding frequencies 

    """
    # Calculate window size in samples and window size x coefs for overlap and zero-pad
    window_size = int(epochs.info["sfreq"]*window_duration)
    n_overlap = int(window_size*window_overlap)
    n_zeropad = int(window_size*zero_padding)

    # N of samples from signals equals to window size
    n_per_seg = window_size

    # N of samples for FFT equals N of samples + zero-padding samples
    n_fft = n_per_seg + n_zeropad

    # Calculate PSD with Welch's method
    spectrum = epochs.compute_psd(method=method, fmin = fminmax[0], fmax = fminmax[1],
                                  n_fft=n_fft, n_per_seg=n_per_seg, n_overlap=n_overlap,
                                  window=window, tmin=tminmax[0], tmax=tminmax[1],
                                  verbose=False)
    
    psds, freqs = spectrum.get_data(return_freqs = True)

    # Unit conversion from V^2/Hz to uV^2/Hz
    psds = psds*1e12

    # If true, print all the parameters involved in PSD calculation
    if verbose == True:
        print("---\nPSD ({}) calculation\n".format(method))
        print(spectrum)
        print('Time period:', str(tminmax))
        print('Window type:', window)
        print('Window size:', window_size)
        print('Overlap:', n_overlap)
        print('Zero-padding:', n_zeropad)
        print('\nSamples per segment:', n_per_seg)
        print('Samples for FFT:', n_fft)
        print('Frequency resolution:', freqs[1]-freqs[0], 'Hz')

    # If true, plot average PSD for all epochs and channels with channel PSDs
    if plot == True:
        plt.figure(figsize=(5,3), dpi=100)
        plt.plot(freqs,np.transpose(psds.mean(axis=(0))),color='black',alpha=0.1)
        plt.plot(freqs,psds.mean(axis=(0, 1)),color='blue',alpha=1)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('PSD (uV\u00b2/Hz)')
        plt.title("PSD,{} ({})".format(method, subjectname))
        plt.xlim(fminmax)
        plt.ylim(0,None)
        plt.grid(linewidth=0.2)
        plt.show()

    return [psds,freqs]

def bandpower_per_channel(psds, freqs, band, b_name, subjectname, epochs,  
                          ln_normalization = False, verbose = True):
    """
    Find frequency band in interest for all the channels. 

    Parameters
    ----------
    psds: An array for power spectrum density values 
    freqs: An array for corresponding frequencies
    band: A list of lower and higher frequency for the frequency band in interest
    b_name: A string for frequency band in interest
    subjectname: A string for subject's name
    epochs: Epochs-type (MNE-Python) EEG file

    Returns
    --------
    psds_band_mean_ch : An array for a frequency band power values for all the channels 
    
    """

    # Average all epochs together for each channels' PSD values
    psds_per_ch = psds.mean(axis=(0))

    # Pick only PSD values which are within the frequency band of interest
    idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])
    psds_band_per_ch = psds_per_ch[:, idx_band]

    # Average the PSD values within the band together to get bandpower for each channel
    bp_per_ch = psds_band_per_ch.mean(axis=(1))

    # If true, normalize the BP with natural logarithm transform
    if ln_normalization == True:
        bp_per_ch = np.log(bp_per_ch)
    
    if verbose == True:
        print("Finding bandpower within {} Hz with Ln normalization set to {}".format(band, str(ln_normalization)))
    
    return bp_per_ch

def find_ind_band(spectrum, freqs, freq_interest=[4, 8], bw_size=4):
    # Get indexes of band of interest
    freq_interest_idx = np.where(np.logical_and(freqs>=freq_interest[0],
                        freqs<=freq_interest[1]))
    
    # Find maximum amplitude (peak width) in that bandwidth
    pw = np.max(spectrum[freq_interest_idx])

    # Find center frequency index and value where the peak is
    cf_idx = np.where(spectrum == pw)
    cf = float(freqs[cf_idx])
    
    # Get bandwidth range for the band np.round(bw[0], 4)
    bw = [np.round(cf-bw_size/2, 4), np.round(cf+bw_size/2, 4)]

    # Find individual bandpower indexes based on the binsize
    bp_idx = np.logical_and(freqs>=bw[0], freqs<=bw[1])

    # Average the PSD values in these indexes together to get bandpower
    abs_bp = spectrum[bp_idx].mean()

    # Calculate relative bandpower
    rel_bp = abs_bp / spectrum.mean()

    return cf, pw, bw, abs_bp, rel_bp