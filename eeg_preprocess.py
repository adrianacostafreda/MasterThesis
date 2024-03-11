import random
import mne
import pyxdf
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from mne.preprocessing import ICA
import os
from pyxdf import match_streaminfos, resolve_streams
from mnelab.io.xdf import read_raw_xdf
import autoreject
from autoreject import AutoReject

def get_raw_from_xdf(xdf_file_path: str) -> mne.io.Raw:
    """This function loads an XDF EEG file, and returns a mne.io.Raw object
    
    Parameters
    ----------
    xdf_file_path : str
        Path to the XDF file. 
    
    """
    streams = resolve_streams(xdf_file_path)
    stream_id = match_streaminfos(streams, [{"type":"EEG"}])
    raw = read_raw_xdf(xdf_file_path, stream_ids = [1,2], fs_new=500)

    # Drop channels which are not needed for the analysis
    raw.drop_channels(["ACC_X","ACC_Y","ACC_Z","Trigger_0"])

    # Average
    raw.set_eeg_reference()
    
    return raw

def set_montage(raw):
    
    dic = {"Fp1":"AFp1", "Fz": "AFF1h", "F3": "AF7", "F7": "AFF5h", "F9":"FT7", "FC5": "FC5", "FC1":"FC3", "C3":"FCC3h",
           "T7":"FFC1h", "CP5": "FCC1h", "CP1": "CCP3h", "Pz":"CCP1h", "P3":"CP1", "P7": "CP3", "P9":"CPP3h", "O1":"P1",
           "Oz":"AFp2", "O2":"AFF2h", "P10":"AF8", "P8":"AFF6h", "P4":"FT8", "CP2":"FC6", "CP6":"FC4", "T8":"FCC4h",
           "C4":"FFC2h", "Cz":"FCC2h", "FC2":"CCP4h", "FC6": "CCP2h", "F10":"CP2", "F8":"CP4", "F4":"CPP4h", "Fp2":"P2"}
    raw.rename_channels(dic)
    print(raw.info["ch_names"])

    cap_montage = mne.channels.make_standard_montage("brainproducts-RNP-BA-128")

    # Use the preloaded montage
    raw_montage = raw.set_montage(cap_montage)
    # raw_montage.plot_sensors(show_names = True)
    # plt.show()
    return raw_montage

def preprocess(raw: mne.io.Raw) -> None:
    """Preprocess EEG data: filter (bandpass filter between 0.1 and 30 Hz)
    
    Source : https://neuraldatascience.io/7-eeg/erp_filtering 

    """
    raw.filter(l_freq = 0.1, h_freq = 30, verbose = False)
    return

def repair_artifacts_ICA(raw_montage):
    
    """
    Use Independent Component Analysis for artifact repair
    """

    """ # Filter settings
    ica_low_cut = 1.0 # For ICA, we filter out more low-frequency power
    hi_cut  = 30

    raw_ica = raw_montage.copy().filter(ica_low_cut, hi_cut) """

    # Break raw data into 1 s epochs
    tstep = 1.0
    events_ica = mne.make_fixed_length_events(raw_montage, duration=tstep)
    epochs_ica = mne.Epochs(raw_montage, events_ica,
                    tmin=0.0, tmax=tstep,
                    baseline=None,
                    preload=True)

    ar = AutoReject(n_interpolate=[1, 2, 4],
            random_state=42,
            picks=mne.pick_types(epochs_ica.info, 
                                eeg=True,
                                ),
                n_jobs=-1, 
                verbose=False
                )

    ar.fit(epochs_ica)

    reject_log = ar.get_reject_log(epochs_ica)

    # ICA parameters
    random_state = 42   # ensures ICA is reproducible each time it's run
    ica_n_components = .99     # Specify n_components as a decimal to set % explained variance

    # Fit ICA
    ica = mne.preprocessing.ICA(n_components=ica_n_components,
                            random_state=random_state,
                            )
    ica.fit(epochs_ica[~reject_log.bad_epochs], decim=3)

    ica.plot_sources(epochs_ica, block = True)

    #ica.plot_components()
    #ica.plot_properties(epochs_ica, picks=range(0, ica.n_components_), psd_args={'fmax': hi_cut})
    #plt.show()

    # Save ICA
    """
    We will save the ICA object, which includes the bads attribute which indicates which components to reject as ocular artifacts
    """
    # ica.save(data_path + fname +"-ica.fif", overwrite = True )
    
    # picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
    # eog_average = mne.preprocessing.create_eog_epochs(epochs_ica, picks = picks).average()
    # eog_epochs = mne.preprocessing.create_eog_epochs(epochs_ica)

    ica.exclude = []
    num_excl = 0
    max_ic = 2
    z_thresh = 3.5
    z_step = .05

    while num_excl < max_ic:
        eog_indices, eog_scores = ica.find_bads_eog(epochs_ica,
                                                ch_name=["AFp1", "AF7", "AF8", "AFp2"], 
                                                threshold=z_thresh
                                                )
        num_excl = len(eog_indices)
        z_thresh -= z_step # won't impact things if num_excl is ≥ n_max_eog 

    # eog_inds, scores = ica.find_bads_eog(eog_epochs)
    ica.plot_scores(eog_scores,exclude=eog_indices)
    plt.show()

    return (ica, eog_indices)

def ica_template_match(icas,eog_indices):
    mne.preprocessing.corrmap(icas, template =(0,eog_indices[0]))
    plt.show()


def repair_artifacts_ICA_2(raw_montage):
    """
    Use Independent Component Analysis for artifact repair
    """

    # ICA parameters
    random_state = 42   # ensures ICA is reproducible each time it's run
    ica_n_components = .99     # Specify n_components as a decimal to set % explained variance

    # Fit ICA
    ica = mne.preprocessing.ICA(n_components=ica_n_components,
                            random_state=random_state,
                            )
    ica.fit(raw_montage)

    ica.plot_sources(raw_montage, block=True)
    plt.show()
    #ica.plot_components()
    #ica.plot_properties(epochs_ica, picks=range(0, ica.n_components_), psd_args={'fmax': hi_cut})
    #plt.show()

    # Save ICA
    """
    We will save the ICA object, which includes the bads attribute which indicates which components to reject as ocular artifacts
    """
    # ica.save(data_path + fname +"-ica.fif", overwrite = True )
        
def identify_bad_ica(raw, ica, epochs_ica, raws, icas):

    """
    Identify EOG Artifacts from ICA Components: the "find_bads_eog()" function computes correlations between each IC component 
    and channels that the researcher has designed as EOG channels. The present data set does not have EOG electrodes. However, we
    use electrodes close to the eyes which will detect horizontal movements
    """

    # Find the right threshold
    ica.exclude = []
    """ num_excl = 0
    max_ic = 2
    z_thresh = 3.5
    z_step = .05

    while num_excl < max_ic:
        eog_indices, eog_scores = ica.find_bads_eog(epochs_ica,
                                                ch_name=['AFp1', 'AF7', 'AFp2', 'AF8'], 
                                                threshold=z_thresh
                                                )
        num_excl = len(eog_indices)
        z_thresh -= z_step # won't impact things if num_excl is ≥ n_max_eog 

    # assign the bad EOG components to the ICA.exclude attribute so they can be removed later
    ica.exclude = eog_indices

    print('Final z threshold = ' + str(round(z_thresh, 2))) """
    # Barplot of ICA component 
    #ica.plot_scores(eog_scores)

    # plot diagnostics 
    #ica.plot_properties(raw,picks=eog_indices)

    #plot ICs applied to raw data
    #ica.plot_sources(raw, show_scrollbars = False)


    # Using a simulated channel to select ICA components 
    """ If you do not have an EOG channel, "fins_bads_eog" has a ch_name parameter that you can use as a proxy for EOG. A single channel
    can be used or create a bipolar reference from frontal EEG sensors and use that as a virtual EOG channel. This carries the risk that frontal
    EEG channels only reflect EOG or not care about the brain dynamic signals"""

    # Selecting ICA components using template matching
    """It is possible to manually select an IC for exclusion on one subject, and then use that component as a template for 
    selecting which ICAa to exclude from other subjects"""

    """ from mne.processing import create_eog_epochs 
    eog_average = create_eog_epochs(raw).average()
    indexes, scores = ica.find_bads_eog(eog_average)
    
    ica.plot_sources(eog_average,exclude = indexes) """

def plot_spectrum(raw):
    """This function is called if we want to plot lines in the frequency points after fig = spectrum.plot()"""
    # Plot the Theta frequency theta
    
    fig = raw.compute_psd().plot(average=True, picks="data", exclude ="bads")
    
    # add some arrows at 60 Hz and its harmonics:
    frequencies = []
    power_spectrum = []

    for ax in fig.axes[0:]:
        freqs = ax.lines[-1].get_xdata()
        psds = ax.lines[-1].get_ydata()
        for freq in (0,4,8,13,30): # Delta, Theta, Alpha, Beta, Gamma
            idx = np.searchsorted(freqs, freq)
            frequencies.append(freqs[idx])
            power_spectrum.append(psds[idx])
            plt.axvline(x=freqs[idx], color="b")
            
    plt.show()

def set_trigger4(raw):
    """Define an annotation for the Trigger 4"""

    annot = raw.annotations
    onset_trigg4 = annot.onset[8]-15-40
    channels = list(raw.info["ch_names"])

    # Use list comprehensions to find indices of each string
    index_AFp1 = [i for i, chann in enumerate(channels) if chann == 'AFp1']
    index_AFF1h = [i for i, chann in enumerate(channels) if chann == 'AFF1h']
    index_FFC1h = [i for i, chann in enumerate(channels) if chann == 'FFC1h']
    index_FCC1h = [i for i, chann in enumerate(channels) if chann == 'FCC1h']
    index_AFp2 = [i for i, chann in enumerate(channels) if chann == 'AFp2']
    index_AFF2h = [i for i, chann in enumerate(channels) if chann == 'AFF2h']
    index_FFC2h = [i for i, chann in enumerate(channels) if chann == 'FFC2h']
    index_FCC2h = [i for i, chann in enumerate(channels) if chann == 'FCC2h']

    #indices_to_index = [index_Fz[0], index_Cz[0], index_Pz[0]]
    #indexed_elements = [channels[idx] for idx, _ in enumerate(channels) if idx in indices_to_index]

    #trigger4 = mne.Annotations(onset_trigg4, 48, "4", orig_time = None, ch_names = [indexed_elements])
    trigger4 = mne.Annotations(onset_trigg4, 48, "4", orig_time = None)
    new_annotations = annot + trigger4
    raw.set_annotations(new_annotations)

def epochs_from_raw(raw):
    """
    Returns a mne.Epochs object created from the annotated mne.io.Raw object.
    The length of the epochs is set to the duration of each trigger corresponding to the N-Back test

    Events are created around the events of interest which correspond to 4,5,6,7

    """

    events, _ = mne.events_from_annotations(raw) # Create events from existing annotations
    events = events[np.in1d(events[:,2], (4, 5, 6,7)), : ]
    event_id = {"Trigger 4": 4, "Trigger 5": 5, "Trigger 6": 6, "Trigger 7": 7}

    # Using this event array, we take continuous data and create epochs ranging from -0.5 seconds before to 60 seconds after each event for 5,6,7 and 48 seconds for 4 
    # In other words, an epoch comprises data from -0.5 to 60 seconds around 5, 6, 7 events. For event 4, the epoch will comprise data from -0.5 to 48 seconds. 
    # We will consider Fz, Cz, Pz channels corresponding to the mid frontal line

    tmin_4, tmax_4 = -0.1, 48
    event4 = mne.pick_events(events, include=[4])
    event4_dict ={"Trigger 4": 4}

    epochs_4 = mne.Epochs(
        raw,
        event4,
        event4_dict,
        tmin_4,
        tmax_4,
        #picks=("Fz", "Cz", "Pz"),
        baseline=None,
        preload=True
    )

    tmin, tmax = -0.1, 60
    event_567 = mne.pick_events(events, include =[5,6,7])
    event_567_dict = {
        "Trigger 5": 5,
        "Trigger 6": 6,
        "Trigger 7":7 }

    epochs_567 = mne.Epochs(
        raw,
        event_567, 
        event_567_dict,
        tmin,
        tmax,
        #picks=("Fz", "Cz", "Pz"),
        baseline=None,
        preload=True
    )

    epochs_5 = epochs_567["Trigger 5"]
    epochs_6 = epochs_567["Trigger 6"]
    epochs_7 = epochs_567["Trigger 7"]

    #epochs_4.plot(n_epochs = 1000, events = event4, event_id = event4_dict, scalings = 2e-4)
    #epochs_567["Trigger 5"].plot(n_epochs =1000, events=True, scalings = 2e-4)
    #epochs_567["Trigger 6"].plot(n_epochs =1000, events=True, scalings = 2e-4)
    #epochs_567["Trigger 7"].plot(n_epochs =1000, events=True, scalings = 2e-4)

    return (epochs_4, epochs_5, epochs_6, epochs_7)

def ica_correction_to_epochs(raw_montage, datapath, fname, epochs_4, epochs_5, epochs_6, epochs_7):
    
    # raw.set_montage("easycap-M1")
    # easycap_montage = mne.channels.make_standard_montage("easycap-M1")
    # raw.set_montage(easycap_montage)

    """Apply ICA correction to epochs"""

    # read previously-saved ICA decomposition
    ica = mne.preprocessing.read_ica(datapath + fname + "-ica.fif")

    # apply the ICA decomposition to the epochs
    epochs4_postica = ica.apply(epochs_4.copy())
    epochs5_postica = ica.apply(epochs_5.copy())
    epochs6_postica = ica.apply(epochs_6.copy())
    epochs7_postica = ica.apply(epochs_7.copy())
    
    fix, ax = plt.subplots(nrows = 1, ncols = 2, figsize=(20,5))
    epochs_4.average().plot(axes = ax[0], ylim = [-11,10], show = False, spatial_colors=True)
    epochs4_postica.average().plot(axes = ax[1], ylim = [-11,10], show = False, spatial_colors=True)
    plt.tight_layout()
    plt.show()

    fix, ax = plt.subplots(nrows = 1, ncols = 2, figsize=(20,5))
    epochs_5.average().plot(axes = ax[0], ylim = [-11,10], show = False, spatial_colors=True)
    epochs5_postica.average().plot(axes = ax[1], ylim = [-11,10], show = False, spatial_colors=True)
    plt.tight_layout()
    plt.show()

    fix, ax = plt.subplots(nrows = 1, ncols = 2, figsize=(20,5))
    epochs_6.average().plot(axes = ax[0], ylim = [-11,10], show = False, spatial_colors=True)
    epochs6_postica.average().plot(axes = ax[1], ylim = [-11,10], show = False, spatial_colors=True)
    plt.tight_layout()
    plt.show()

    fix, ax = plt.subplots(nrows = 1, ncols = 2, figsize=(20,5))
    epochs_7.average().plot(axes = ax[0], ylim = [-11,10], show = False, spatial_colors=True)
    epochs7_postica.average().plot(axes = ax[1], ylim = [-11,10], show = False, spatial_colors=True)
    plt.tight_layout()
    plt.show()

    """ # AutoReject for Final Data Cleaning
    ar = AutoReject(n_interpolate=[1, 2, 4], 
                    random_state=42,
                    n_jobs=-1,
                    verbose=False)
    
    epochs4_clean, reject_log_clean_epoch4 = ar.fit_transform(epochs4_postica,return_log = True)
    epochs5_clean, reject_log_clean_epoch5 = ar.fit_transform(epochs5_postica,return_log = True)
    epochs6_clean, reject_log_clean_epoch4 = ar.fit_transform(epochs6_postica,return_log = True)
    epochs7_clean, reject_log_clean_epoch4 = ar.fit_transform(epochs7_postica,return_log = True) """

def regression_baseline_correction(raw, epochs_4, epochs_5, epochs_6, epochs_7):
    """
    Compare traditional baseline correction (adding or subtracting a scalar amount from every timepoint in an epoch)
    to a regression-based approach to baseline correction (allows the effect of the baseline period to vary by timepoint)
    """
    baseline_tmin, baseline_tmax = None, 0 # None takes the first timepoint
    
    """
    Traditional Baseline: we will baseline correct the data the traditional way. We average epochs within each condition and 
    subtract the condition specific baseline
    """
    baseline = (baseline_tmin, baseline_tmax)
    trad_trigger4 = epochs_4.average().apply_baseline(baseline)
    trad_trigger5 = epochs_5.average().apply_baseline(baseline)
    trad_trigger6 = epochs_6.average().apply_baseline(baseline)
    trad_trigger7 = epochs_7.average().apply_baseline(baseline)

    """
    Regression-base baselining: We will use mne.stats.linear_regression(), which needs a design matrix to represent the regression 
    predictors. We will use four predictors:
        - one for each experimental condition
        - one for the effect of the baseline
        - one which is the interaction between the baseline
        - one of the conditions (account for any heterogeneity of the effect of baseline between two conditions)
    """
    
    """ trigger4_predictor = epochs_4.events[:, 1] == epochs_4.event_id["Trigger 4"]
    trigger5_predictor = epochs_5.events[:, 1] == epochs_5.event_id["Trigger 5"]
    trigger6_predictor = epochs_6.events[:, 1] == epochs_6.event_id["Trigger 6"]
    trigger7_predictor = epochs_7.events[:, 1] == epochs_7.event_id["Trigger 7"]

    baseline_predictor = (
        epochs_5.copy()
        .crop(*baseline)
        .get_data(copy=False) # Convert to numpy array
        .mean(axis=-1) # average across timepoints
        .squeeze() 
    )
    baseline_predictor *= 1e6 # convert V -> uV

    design_matrix = np.vstack(
        [
            trigger5_predictor,
            trigger5_predictor,
            trigger6_predictor,
            trigger7_predictor,
            baseline_predictor,
            baseline_predictor*trigger5_predictor
        ]
    ).T

    reg_model = mne.stats.linear_regression(
        epochs_5, design_matrix, names = ["Trigger 4", "Trigger 5", "Trigger 6", "Trigger 7", "baseline", "baseline:Trigger6"]
    ) """

def evoked_data(epochs_4, epochs_567):

    """Evoked data are obtained by averaging epochs. Typically, an evoked object is constructed for each subject and each condition."""

    evoked_list = []
    evoked_4 = epochs_4.average()
    evoked_list.append(evoked_4)
    evoked_5 = epochs_567["Trigger 5"].average()
    evoked_list.append(evoked_5)
    evoked_6 = epochs_567["Trigger 6"].average()
    evoked_list.append(evoked_6)
    evoked_7 = epochs_567["Trigger 7"].average()
    evoked_list.append(evoked_7)

    print(f'Epochs baseline: {epochs_4.baseline}')
    print(f'Evoked baseline: {evoked_4.baseline}')
    print(f'Epochs baseline: {epochs_567.baseline}')
    print(f'Evoked baseline: {evoked_5.baseline}')
    print(f'Epochs baseline: {epochs_567.baseline}')
    print(f'Evoked baseline: {evoked_6.baseline}')
    print(f'Epochs baseline: {epochs_567.baseline}')
    print(f'Evoked baseline: {evoked_7.baseline}')

    # Create subplots 
    fig,axes = plt.subplots(nrows = 2, ncols=2, figsize=(10,8))

    #Plot each evoked in a subplot
    for i, evoked in enumerate(evoked_list):
        row = i // 2
        col = i % 2
        evoked.plot(axes=axes[row,col],spatial_colors=True, gfp=True)
        axes[row,col].set_title("Trigger {}".format(i+1))
    plt.tight_layout()
    plt.show()

def test_xdf_files_reading():
    """
    Tests if the XDF files can be read as RAW objects
    """
    paths = list()
    failed = list()

    # Read the data hc baseline fnirs
    path_hc_baseline_eeg = "H:\\Dokumenter\\data_processing\\data_eeg\\patients\\followup\\"
    folder_hc_baseline_eeg = os.fsencode(path_hc_baseline_eeg)

    for file in os.listdir(folder_hc_baseline_eeg):
        filename = os.fsdecode(file)
        if filename.endswith(".xdf"):
            fname = path_hc_baseline_eeg + filename
            paths.append(fname)
    
    for path in tqdm(paths):
        try:
            _ = get_raw_from_xdf(path)
        except:
            failed.append(path)
    
    print("Failed for file(s) located at: ")
    for path in failed:
        print(path)
    
    return len(failed) == 0

#assert test_xdf_files_reading()

mne.set_config("MNE_BROWSER_BACKEND", "qt")
paths = list()

# Read the data hc baseline fnirs
path_hc_baseline_eeg = "H:\\Dokumenter\\data_processing\\data_eeg\\healthy_controls\\baseline\\try\\"
#path_hc_baseline_eeg = "//Users//adriana//Library//CloudStorage//OneDrive-Personal//Documentos//AA_MASTER//AA_Q4//TFM//eeg_processing//visual_studio_eeg//try//"
folder_hc_baseline_eeg = os.fsencode(path_hc_baseline_eeg)

# Create list where I will append the epochs for each of the files
raw_hc_baseline_trigger4 = []
raw_hc_baseline_trigger5 = []
raw_hc_baseline_trigger6 = []
raw_hc_baseline_trigger7 = []

raws = list()
icas = list()

for file in os.listdir(folder_hc_baseline_eeg):
    filename = os.fsdecode(file)
    if filename.endswith(".xdf"):
        fname = path_hc_baseline_eeg + filename
        paths.append(fname)

for path in paths:
    print("\nNow working with file", path, "\n")

    raw = get_raw_from_xdf(path)
    raw = raw.copy()
    preprocess(raw)
    filt_raw = raw.copy().filter(l_freq = 1, h_freq = 40)
    raw_montage = set_montage(filt_raw)

    # raw.load_data()
    # raw.plot(block=True)
    # plot_spectrum(raw)

    returns = repair_artifacts_ICA(raw_montage)
    icas.append(returns[0])
    print(icas)
    
    set_trigger4(raw_montage)
    triggers = epochs_from_raw(raw_montage)
    
    raw_hc_baseline_trigger4.append(triggers[0])
    raw_hc_baseline_trigger5.append(triggers[1])
    raw_hc_baseline_trigger6.append(triggers[2])
    raw_hc_baseline_trigger7.append(triggers[3])

    # regression_baseline_correction(raw, triggers[0], triggers[1], triggers[2], triggers[3])
    # ica_correction_to_epochs(raw_montage, path_hc_baseline_eeg, filename, triggers[0], triggers[1], triggers[2], triggers[3])