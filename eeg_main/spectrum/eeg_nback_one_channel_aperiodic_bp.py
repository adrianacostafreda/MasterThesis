# Import packages
import os, mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fooof import FOOOF
from fooof.plts.spectra import plot_spectrum, plot_spectrum_shading

from IPython.display import display

# Set default directory
os.chdir("H:\Dokumenter\GitHub\MasterThesis\.venv")

# Import functions
import signal_processing.spectrum as spectrum
import basic.arrange_files as arrange

# Folder where to get the clean epochs files
clean_folder = "H:\\Dokumenter\\data_acquisition\\data_eeg\\clean\\"

# Folder where to save the results
results_foldername = "H:\\Dokumenter\\data_processing\\Results\\"

# Sub-folder for the experiment (i.e. timepoint or group) and its acronym
exp_folder = 'baseline'
exp_condition = '3_back'

# Brain regions and their channels
ch = 'FCC1h'

# Power spectra estimation parameters
psd_params = dict(method='welch', fminmax=[1, 30], window='hamming', window_duration=2,
                  window_overlap=0.25, zero_padding=39)

# FOOOF (specparam) model parameters
fooof_params = dict(peak_width_limits=[1,12], max_n_peaks=float('inf'), min_peak_height=0.225,
                    peak_threshold=2.0, aperiodic_mode='fixed')

# Band power of interest
bands = {'Theta' : [4, 8]}

# Flattened spectra amplitude scale (linear, log)
flat_spectr_scale = 'linear'

# Plot more information on the model fit plots or not; and save these plots or not
plot_rich = True
savefig = True

# Event names (i.e. different stimuli) within the epochs
#event_list = ['0-Back', '1-Back', '2-Back', '3-Back']

# Get directories of clean EEG files and set export directory
dir_inprogress = os.path.join(clean_folder, exp_folder, exp_condition)
file_dirs, subject_names = arrange.read_files(dir_inprogress, "_clean-epo.fif")

# Pre-create results folders and dataframe
arrange.create_results_folders(exp_folder=exp_folder, results_folder=results_foldername, fooof=True)
df_ch = pd.DataFrame()

# Go through all the files (subjects) in the folder
for i in range(len(file_dirs)):

    # Read the clean data from the disk
    epochs = mne.read_epochs(fname='{}/{}_clean-epo.fif'.format(dir_inprogress, subject_names[i]),
                                                                verbose=False)
    

    df_ch_ev = pd.DataFrame()

     # Calculate Welch's power spectrum density (FFT) for the mean post-event
    [psds, freqs] = spectrum.calculate_psd(epochs, subject_names[i], **psd_params, verbose=True, plot=False)
        
    # Average all epochs and channels together -> (freq bins,) shape
    if i == 0:
        psds_allch = np.zeros(shape=(len(file_dirs), len(freqs)))
    psds_allch[i] = psds.mean(axis=(0, 1))

    # Average all epochs together for each channel and also for each region
    psds = psds.mean(axis=(0))
    df_psds_ch = arrange.array_to_df(subject_names[i], epochs, psds).\
                            reset_index().drop(columns='Subject')

    # Choose only channel of interest data
    psds_temp = df_psds_ch[ch].to_numpy()

    # Fit the spectrums with FOOOF
    fm = FOOOF(**fooof_params, verbose=True)
    fm.fit(freqs, psds_temp, psd_params['fminmax'])

    # Log-linear conversion based on the chosen amplitude scale
    if flat_spectr_scale == 'linear':
        flatten_spectrum = 10 ** fm._spectrum_flat
        flat_spectr_ylabel = 'Amplitude (uV\u00b2/Hz)'
    elif flat_spectr_scale == 'log':
        flatten_spectrum = fm._spectrum_flat
        flat_spectr_ylabel = 'Log-normalised amplitude'

    # Find individual theta band parameters
    abs_bp, rel_bp = spectrum.find_bp(flatten_spectrum, freqs, bands['Theta'])

    ### PLOTTING

    # Set plot styles
    data_kwargs = {'color' : 'black', 'linewidth' : 1.4, 'label' : 'Original'}
    model_kwargs = {'color' : 'red', 'linewidth' : 1.4, 'alpha' : 0.75, 'label' : 'Full model'}
    aperiodic_kwargs = {'color' : 'blue', 'linewidth' : 1.4, 'alpha' : 0.75,
                            'linestyle' : 'dashed', 'label' : 'Aperiodic model'}
    flat_kwargs = {'color' : 'black', 'linewidth' : 1.4}
    hvline_kwargs = {'color' : 'blue', 'linewidth' : 1.0, 'linestyle' : 'dashed', 'alpha' : 0.75}

    # Plot power spectrum model + aperiodic fit
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 4), dpi=100)
    plot_spectrum(fm.freqs, fm.power_spectrum,
                ax=axs[0], **data_kwargs)
    plot_spectrum(fm.freqs, fm.fooofed_spectrum_,
                ax=axs[0], **model_kwargs)
    plot_spectrum(fm.freqs, fm._ap_fit,
                ax=axs[0], **aperiodic_kwargs)
    axs[0].set_xlim(psd_params['fminmax'])
    axs[0].grid(linewidth=0.2)
    axs[0].set_xlabel('Frequency (Hz)')
    axs[0].set_ylabel('Log-normalised power (log$_{10}$[µV\u00b2/Hz])')
    axs[0].set_title('Spectrum model fit')
    axs[0].legend()

    # Flattened spectrum plot (i.e., minus aperiodic fit)
    plot_spectrum_shading(fm.freqs, flatten_spectrum,
                ax=axs[1], shades=bands['Theta'], shade_colors='green', **flat_kwargs)
    #axs[1].vlines(bands['Theta'], ymin=axs[1].get_ylim()[0], ymax=axs[1].get_ylim()[1])
    axs[1].set_xlim(psd_params['fminmax'])
    axs[1].grid(linewidth=0.2)
    axs[1].set_xlabel('Frequency (Hz)')
    axs[1].set_ylabel(flat_spectr_ylabel)
    axs[1].set_title('Flattened spectrum')
    

    # If true, plot all the exported variables on the plots
    if plot_rich == True:
        axs[0].annotate('Error: ' + str(np.round(fm.get_params('error'), 4)) +
                        '\nR\u00b2: ' + str(np.round(fm.get_params('r_squared'), 4)),
                        (0.1, 0.16), xycoords='figure fraction', color='red', fontsize=8.5)
        axs[0].annotate('Exponent: ' + str(np.round(fm.get_params('aperiodic_params','exponent'), 4)) +
                        '\nOffset: ' + str(np.round(fm.get_params('aperiodic_params','offset'), 4)),
                        (0.19, 0.16), xycoords='figure fraction', color='blue', fontsize=8.5)
        axs[1].annotate('Absolute theta BP: '+str(np.round(abs_bp, 4))+'\nRelative theta BP: '+str(np.round(rel_bp, 4)),
                        (0.69, 0.16), xycoords='figure fraction', color='green', fontsize=8.5)


    plt.suptitle('PSD at {} ({})'.format(ch, subject_names[i]))
    plt.tight_layout()
    if savefig == True:
        plt.savefig(fname='{}/{}/FOOOF/{}/{}_{}_mean_post_event_PSD.png'.format(results_foldername, exp_folder,
                                                                        exp_condition, subject_names[i],
                                                                        ch), dpi=300)
    #plt.show()


    ### EXPORTING

    # Add model parameters to dataframe for mean post-event
    df_ch_ev.loc[i, 'Exponent'] = fm.get_params('aperiodic_params','exponent')
    df_ch_ev.loc[i, 'Offset'] = fm.get_params('aperiodic_params','offset')
    df_ch_ev.loc[i, '{} absolute power'.format(list(bands.keys())[0])] = abs_bp
    df_ch_ev.loc[i, '{} relative power'.format(list(bands.keys())[0])] = rel_bp
    df_ch_ev.loc[i, 'R_2'] = fm.get_params('r_squared')
    df_ch_ev.loc[i, 'Error'] = fm.get_params('error')
    df_ch_ev['Channel'] = ch
    #df_ch_ev['Type'] = 'Mean post-event'
    df_ch_ev['Subject'] = subject_names[i]

    # Concatenate to master dataframe for mean post-event
    df_ch = pd.concat([df_ch, df_ch_ev])


# Reorder the channels and reset index
df_ch = df_ch[['Subject', 'Channel', 'Exponent', 'Offset',
               '{} absolute power'.format(list(bands.keys())[0]),
               '{} relative power'.format(list(bands.keys())[0]),
               'R_2', 'Error']]
df_ch = df_ch.reset_index(drop=True)

# Export results for post-event data
df_ch.to_excel('{}/{}/FOOOF/{}/{}_specparam.xlsx'.format(results_foldername, exp_folder, exp_condition, ch))
display(df_ch)



