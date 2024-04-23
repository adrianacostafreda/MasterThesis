import os
import mne
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from IPython.display import display

# Set default directory
os.chdir("H:\Dokumenter\GitHub\MasterThesis\.venv")
mne.set_log_level('error')

# Import functions
import signal_processing.spectrum as spectrum
import basic.arrange_files as arrange

# Folder where to get the clean epochs files
clean_folder = "H:\\Dokumenter\\data_acquisition\\data_eeg\\clean\\"

# Folder where to save the results
results_foldername = "H:\\Dokumenter\\data_processing\\Results\\"

exp_folder = 'healthy_controls'
exp_condition = '0_back'

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
dir_inprogress = os.path.join(clean_folder, exp_folder, exp_condition)
file_dirs, subject_names = arrange.read_files(dir_inprogress,'_clean-epo.fif')

for i in range(len(file_dirs)):
    # Read the clean data from the disk

    epochs = mne.read_epochs(fname='{}/{}_clean-epo.fif'.format(dir_inprogress, subject_names[i]),
                                                                verbose=False)
        
    [psds, freqs] = spectrum.calculate_psd(epochs, subject_names[i], **psd_params,
                                      verbose = True, plot=False)
        
    psd_ch_allbands = spectrum.bandpower_per_channel(psds, freqs, [b_freqs[0][0], b_freqs[-1][-1]],
                                                'All bands', subject_names[i], epochs)
        
    # Find power for all bands and add to dataframe 
    for j in range(len(b_names)):
        if i == 0:

            globals()["df_psd_" + b_names[j]] = pd.DataFrame()
            globals()["df_rel_psd_"+b_names[j]] = pd.DataFrame()
        
        if j==0:
            vlim = [float('inf'), 0]

        # Divide the PSD to frequency band bins and calculate absolute and relative bandpowers
        globals()["psd_ch_" + b_names[j]] = spectrum.bandpower_per_channel(psds, freqs, b_freqs[j],
                                                b_names[j], subject_names[i], epochs)
        globals()["rel_psd_ch_" + b_names[j]] = globals()["psd_ch_" + b_names[j]]/psd_ch_allbands


        # Convert the array to dataframe and concatenate it to dataframe including the previous subjects
        globals()["temp_df_psd_"+b_names[j]] = arrange.array_to_df(subject_names[i], epochs,
                                                                   globals()["psd_ch_"+b_names[j]])
        globals()["df_psd_"+b_names[j]] = pd.concat([globals()["df_psd_"+b_names[j]],
                                                     globals()["temp_df_psd_"+b_names[j]]])
        globals()["temp_df_rel_psd_"+b_names[j]] = arrange.array_to_df(subject_names[i], epochs,
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
arrange.create_results_folders(results_folder=results_foldername, exp_folder=exp_folder,
                               abs_psd=True, rel_psd=True)

# Export power spectra data for each band
for band in b_names:
    # Save the PSD values for each channel for each band in Excel format
    globals()["df_psd_"+band].to_excel('{}/{}/Absolute PSD/channels/{}_psd_{}.xlsx'.format(results_foldername, exp_folder, exp_condition, band))
    globals()["df_rel_psd_"+band].to_excel('{}/{}/Relative PSD/channels/{}_psd_{}.xlsx'.format(results_foldername, exp_folder, exp_condition, band))

    # Find regional band powers and display and save them to Excel
    df_psd_band_reg = arrange.df_channels_to_regions(globals()["df_psd_"+band], brain_regions)
    df_psd_band_reg.to_excel('{}/{}/Absolute PSD/regions/{}_psd_{}.xlsx'.format(results_foldername, exp_folder, exp_condition, band))
    print('---\nRegional absolute powers in {} band'. format(band))
    display(df_psd_band_reg.head())

    df_psd_band_reg = arrange.df_channels_to_regions(globals()["df_rel_psd_"+band], brain_regions)
    df_psd_band_reg.to_excel('{}/{}/Relative PSD/regions/{}_psd_{}.xlsx'.format(results_foldername, exp_folder, exp_condition, band))
    print('---\nRegional relative powers in {} band'. format(band))
    display(df_psd_band_reg.head())


            
            



        



