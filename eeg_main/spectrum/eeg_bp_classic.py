import os
import mne
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from IPython.display import display

# Set default directory
os.chdir("/Users/adriana/Documents/GitHub/thesis/MasterThesis/")
mne.set_log_level('error')

# Import functions
#import signal_processing.spectrum as spectrum
#import basic.arrange_files as arrange

def read_files(dir_inprogress,filetype,exclude_subjects=[],verbose=True):
    """
    Get all the (EEG) file directories and subject names.

    Parameters
    ----------
    dir_inprogress: A string with directory to look for files
    filetype: A string with the ending of the files we are looking for (e.g. '.xdf')

    Returns
    -------
    file_dirs: A list of strings with file directories for all the (EEG) files
    subject_names: A list of strings with all the corresponding subject names
    """

    file_dirs = []
    subject_names = []

    for file in os.listdir(dir_inprogress):
        if file.endswith(filetype):
            file_dirs.append(os.path.join(dir_inprogress, file))
            #subject_names.append(os.path.join(file).removesuffix(filetype))
            subject_names.append(file[:-len(filetype)])

    try:
        for excl_sub in exclude_subjects:
            for i in range(len(subject_names)):
                if excl_sub in subject_names[i]:
                    if verbose == True:
                        print('EXCLUDED SUBJECT: ',excl_sub,'in',subject_names[i],'at',file_dirs[i])
                    del subject_names[i]
                    del file_dirs[i]
                    break
    except:
        pass
    
    file_dirs = sorted(file_dirs)
    subject_names = sorted(subject_names)

    if verbose == True:
        print("Files in {} read in: {}".format(dir_inprogress,len(file_dirs)))

    return [file_dirs, subject_names]

def array_to_df(subjectname, epochs, array_channels):
    """
    Convert channel-based array to Pandas dataframe with channels' and subjects' names. 

    Parameters
    ----------
    fname: the filename 
    epochs: Epochs-type (MNE-Python) EEG file
    array_channels: An array with values for each channel 

    Returns
    df_channels: A dataframe with values for each channel

    """
    df_channels = pd.DataFrame(array_channels).T

    df_channels.columns = epochs.info.ch_names
    df_channels['Subject'] = subjectname
    df_channels.set_index('Subject', inplace = True)

    return df_channels

def df_channels_to_regions(df_psd_band, brain_regions):
    """
    Average channels together based on the defined brain regions.

    Parameters
    ----------
    df_psd_band: A dataframe with PSD values for each channel per subject
    brain_regions: A dictionary of brain regions and EEG channels which they contain
    drop_cols: List of columns which are not channel PSD data

    Returns
    -------
    df_psd_reg_band: A dataframe with PSD values for each brain region per subject
    """

    df_psd_reg_band = pd.DataFrame()
    for region in brain_regions:
        df_temp = df_psd_band[brain_regions[region]].copy().mean(axis=1)
        df_psd_reg_band = pd.concat([df_psd_reg_band, df_temp], axis=1)
        
    df_psd_reg_band.columns = brain_regions.keys()
    df_psd_reg_band.index.name = 'Subject'

    return df_psd_reg_band

def create_results_folders(exp_folder, exp_condition, exp_condition_nback, results_folder='Results', abs_psd=False,
                           rel_psd=False, fooof = False):
    """
    Dummy way to try to pre-create folders for PSD results before exporting them

    Parameters
    ----------
    exp_folder: A string with a relative directory to experiment folder (e.g. 'Eyes Closed\Baseline')
    """
    if abs_psd == True:
        try:
            os.makedirs(os.path.join('{}/{}/{}/Absolute PSD/channels/{}'.format(results_folder, exp_folder, exp_condition, exp_condition_nback)))
        except FileExistsError:
            pass
        try:
            os.makedirs(os.path.join('{}/{}/{}/Absolute PSD/regions/{}'.format(results_folder, exp_folder, exp_condition, exp_condition_nback)))
        except FileExistsError:
            pass
    
    if rel_psd == True:
        try:
            os.makedirs(os.path.join('{}/{}/{}/Relative PSD/channels/{}'.format(results_folder, exp_folder, exp_condition, exp_condition_nback)))
        except FileExistsError:
            pass
        try:
            os.makedirs(os.path.join('{}/{}/{}/Relative PSD/regions/{}'.format(results_folder, exp_folder, exp_condition,exp_condition_nback)))
        except FileExistsError:
            pass
    
    if fooof== True:
        try:
            os.makedirs(os.path.join('{}/{}/{}/FOOOF/{}'.format(results_folder, exp_folder, exp_condition, exp_condition_nback)))
        except FileExistsError:
            pass
    
    try:
        os.makedirs(os.path.join('{}/{}'.format(results_folder, exp_folder)))
    except FileExistsError:
        pass

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


# Folder where to get the clean epochs files
clean_folder = "/Users/adriana/Documents/DTU/thesis/data_acquisition/"

# Folder where to save the results
results_foldername = "/Users/adriana/Documents/DTU/thesis/data_processing/Results/"

exp_folder = 'healthy_controls'
exp_condition = 'n_back'
exp_condition_nback = '3_back'
exp_condition_nback_num = 3

# Band power names and frequency ranges
b_names = ['Delta', 'Theta', 'Alpha', 'Beta']
b_freqs = [[1, 3.9], [4, 7.9], [8, 12], [12.1, 30]]

# Brain regions and their channels
brain_regions = {'frontal region' : ['AF7', 'AFF5h', 'AFp1', 'AFp2', 'AFF1h', 'AFF2h', 'AF8','AFF6h'],
                 'central region' : ['FFC1h', 'FFC2h', 'FC3', 'FC4', 'FCC3h', 'FCC1h', 'FCC2h', 'FCC4h'],
                 'left temporal' : ['FC5', 'FT7', 'TP7'],
                 'right temporal' : ['FC6', 'FT8', 'TP8'],
                 'parietal region' : ['CCP3h', 'CCP1h', 'CCP2h', 'CCP4h', 'CP1', 'CP2', 'CPP3h', 'CPP4h'],
                 'occipital region' : ['P1', 'P2']}

# Power spectra estimation parameters
psd_params = dict(method='welch', fminmax=[1, 30], window='hamming', window_duration=2,
                  window_overlap=0.25, zero_padding=9)

# Get directories of clean EEG files and set export directory
dir_inprogress = os.path.join(clean_folder, exp_folder, exp_condition, exp_condition_nback)
file_dirs, subject_names = read_files(dir_inprogress,'_clean-epo.fif')

for i in range(len(file_dirs)):
    # Read the clean data from the disk

    epochs = mne.read_epochs(fname='{}/{}_clean-epo.fif'.format(dir_inprogress, subject_names[i]),
                                                                verbose=False)
        
    [psds, freqs] = calculate_psd(epochs, subject_names[i], **psd_params,
                                      verbose = True, plot=False)
        
    psd_ch_allbands = bandpower_per_channel(psds, freqs, [b_freqs[0][0], b_freqs[-1][-1]],
                                                'All bands', subject_names[i], epochs)
        
    # Find power for all bands and add to dataframe 
    for j in range(len(b_names)):
        if i == 0:

            globals()["df_psd_" + b_names[j]] = pd.DataFrame()
            globals()["df_rel_psd_"+b_names[j]] = pd.DataFrame()
        
        if j==0:
            vlim = [float('inf'), 0]

        # Divide the PSD to frequency band bins and calculate absolute and relative bandpowers
        globals()["psd_ch_" + b_names[j]] = bandpower_per_channel(psds, freqs, b_freqs[j],
                                                b_names[j], subject_names[i], epochs)
        globals()["rel_psd_ch_" + b_names[j]] = globals()["psd_ch_" + b_names[j]]/psd_ch_allbands


        # Convert the array to dataframe and concatenate it to dataframe including the previous subjects
        globals()["temp_df_psd_"+b_names[j]] = array_to_df(subject_names[i], epochs,
                                                                   globals()["psd_ch_"+b_names[j]])
        globals()["df_psd_"+b_names[j]] = pd.concat([globals()["df_psd_"+b_names[j]],
                                                     globals()["temp_df_psd_"+b_names[j]]])
        globals()["temp_df_rel_psd_"+b_names[j]] = array_to_df(subject_names[i], epochs,
                                                                       globals()["rel_psd_ch_"+b_names[j]])
        globals()["df_rel_psd_"+b_names[j]] = pd.concat([globals()["df_rel_psd_"+b_names[j]],
                                                         globals()["temp_df_rel_psd_"+b_names[j]]])

        
        # Save the minimum and maximum PSD values as an integer for later colorbar use
        vlim[0] = min([vlim[0], min(globals()["psd_ch_"+b_names[j]])])
        vlim[1] = max([vlim[1], max(globals()["psd_ch_"+b_names[j]])])
    
    # Plot topomaps for all bands
    sns.set_style("white", {'font.family' : ['sans-serif']})
    fig,axs = plt.subplots(nrows=1, ncols=len(b_names), figsize=(10,3))
    fig.suptitle("Frequency topomaps ({})".format(subject_names[i]))
    for topo in range(len(b_names)):
        im,_ = mne.viz.plot_topomap(globals()["psd_ch_"+b_names[topo]], epochs.info, axes=axs[topo],
                                    vlim=vlim, show=False)
        axs[topo].set_title(b_names[topo]+'\n'+str(b_freqs[topo]))
    cbar_ax = fig.add_axes([0.95, 0.35, 0.04, 0.4])
    clb = fig.colorbar(im, cax=cbar_ax)
    clb.ax.set_ylabel('uV\u00b2/Hz')
    #plt.show()

# Pre-create results folders for spectral analysis data
create_results_folders(results_folder=results_foldername, exp_folder=exp_folder, exp_condition=exp_condition,
                       exp_condition_nback=exp_condition_nback, abs_psd=True, rel_psd=True)

# Export power spectra data for each band
for band in b_names:
    # Save the PSD values for each channel for each band in Excel format
    globals()["df_psd_"+band].to_excel('{}/{}/{}/Absolute PSD/channels/{}/{}_psd_{}.xlsx'.format(results_foldername, exp_folder, exp_condition, exp_condition_nback, exp_condition_nback_num, band))
    globals()["df_rel_psd_"+band].to_excel('{}/{}/{}/Relative PSD/channels/{}/{}_psd_{}.xlsx'.format(results_foldername, exp_folder, exp_condition, exp_condition_nback, exp_condition_nback_num, band))

    # Find regional band powers and display and save them to Excel
    df_psd_band_reg = df_channels_to_regions(globals()["df_psd_"+band], brain_regions)
    df_psd_band_reg.to_excel('{}/{}/{}/Absolute PSD/regions/{}/{}_psd_{}.xlsx'.format(results_foldername, exp_folder, exp_condition, exp_condition_nback, exp_condition_nback_num, band))
    print('---\nRegional absolute powers in {} band'. format(band))
    display(df_psd_band_reg.head())

    df_psd_band_reg = df_channels_to_regions(globals()["df_rel_psd_"+band], brain_regions)
    df_psd_band_reg.to_excel('{}/{}/{}/Relative PSD/regions/{}/{}_psd_{}.xlsx'.format(results_foldername, exp_folder, exp_condition, exp_condition_nback, exp_condition_nback_num, band))
    print('---\nRegional relative powers in {} band'. format(band))
    display(df_psd_band_reg.head())


            
            



        



