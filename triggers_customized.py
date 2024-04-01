import mne
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from mne.preprocessing import ICA
import os
from pyxdf import resolve_streams
from mnelab.io.xdf import read_raw_xdf
import autoreject
from autoreject import AutoReject
from pyprep import NoisyChannels

"""

 -----------------------Reading EEG data----------------------------------------------------------------

"""

def get_raw_from_xdf(xdf_file_path: str) -> mne.io.Raw:
    """This function loads an XDF EEG file, and returns a mne.io.Raw object
    
    Parameters
    ----------
    xdf_file_path : str
        Path to the XDF file. 
    
    """
    #streams = resolve_streams(xdf_file_path)
    #stream_id = match_streaminfos(streams, [{"type":"EEG"}])
    raw = read_raw_xdf(xdf_file_path, stream_ids = [1,2], fs_new=500)

    # Drop channels which are not needed for the analysis
    raw.drop_channels(["ACC_X","ACC_Y","ACC_Z","Trigger_0"])
    
    return raw

def set_montage(raw):
    
    dic = {"Fp1":"AFp1", "Fz": "AFF1h", "F3": "AF7", "F7": "AFF5h", "F9":"FT7", "FC5": "FC5", "FC1":"FC3", "C3":"FCC3h",
           "T7":"FFC1h", "CP5": "FCC1h", "CP1": "CCP3h", "Pz":"CCP1h", "P3":"CP1", "P7": "TP7", "P9":"CPP3h", "O1":"P1",
           "Oz":"AFp2", "O2":"AFF2h", "P10":"AF8", "P8":"AFF6h", "P4":"FT8", "CP2":"FC6", "CP6":"FC4", "T8":"FCC4h",
           "C4":"FFC2h", "Cz":"FCC2h", "FC2":"CCP4h", "FC6": "CCP2h", "F10":"CP2", "F8":"P2", "F4":"CPP4h", "Fp2":"TP8"}
    
    raw.rename_channels(dic)

    cap_montage = mne.channels.make_standard_montage("brainproducts-RNP-BA-128")

    # Use the preloaded montage
    raw_montage = raw.set_montage(cap_montage)
    # raw_montage.plot_sensors(show_names = True)
    # plt.show()
    return raw_montage

"""

 -----------------------Preprocessing----------------------------------------------------------------

"""

def filter_eeg(raw: mne.io.Raw) -> None:
    """Preprocess EEG data: filter (bandpass filter between 0.1 and 30 Hz)
    
    Source : https://neuraldatascience.io/7-eeg/erp_filtering 

    """
    raw_filter = raw.filter(l_freq = 0.1, h_freq = 30, verbose = False)
    raw.notch_filter(np.arange(50, 250, 50), verbose=False)
    return raw_filter

def repair_artifacts_ICA(raw,raw_filt):
    
    """
    Use Independent Component Analysis for artifact repair
    """

    # Break raw data into 1 s epochs
    tstep = 1.0
    events_ica = mne.make_fixed_length_events(raw_filt, duration=tstep)
    epochs_ica = mne.Epochs(raw_filt, events_ica,
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

    #ica.plot_sources(epochs_ica, block = True)

    #ica.plot_components()
    #ica.plot_properties(epochs_ica, picks=range(0, ica.n_components_))
    #plt.show()

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

    #ica.plot_scores(eog_scores,exclude=eog_indices)
    #plt.show()

    ica.exclude = list(eog_indices)  # indices chosen based on various plots above
    
    reconst_raw = raw.copy()
    ica.apply(reconst_raw)

    #raw.plot(block=True)
    #reconst_raw.plot(block=True)

    return reconst_raw

"""
----------Part 3: Define Epochs as Events for the N-Back Test-------------------------------

"""

def characterization_trigger_data(raw):
    
    # Dictionary to store trigger data
    hc_trigger_data = {
        '4': {'begin': [], 'end': [], 'duration': []},
        '5': {'begin': [], 'end': [], 'duration': []},
        '6': {'begin': [], 'end': [], 'duration': []},
        '7': {'begin': [], 'end': [], 'duration': []}
    }

    annot = raw.annotations

    first_samp = raw.first_samp

    if raw.info['meas_date'] is None and annot is not None:
                # we need to adjust annotations.onset as when there is no meas
                # date set_annotations considers that the origin of time is the
                # first available sample (ignores first_samp)
        annot.onset -= first_samp / raw.info['sfreq']
        print(annot.onset)
        
    # Iterate over annotations 
    for idx, desc in enumerate(annot.description):
        if desc in hc_trigger_data:
            begin_trigger = annot.onset[idx]
            duration_trigger = annot.duration[idx] + 55
            end_trigger = begin_trigger + duration_trigger

            if end_trigger >= raw.times[-1]:
                end_trigger = raw.times[-1]
            else:
                end_trigger = end_trigger
                
            #store trigger data in the dictionary
            hc_trigger_data[desc]["begin"].append(begin_trigger)
            hc_trigger_data[desc]["end"].append(end_trigger)
            hc_trigger_data[desc]["duration"].append(duration_trigger)
    
    #Determine the start of trigger 4
    for i in hc_trigger_data["5"]["begin"]:
        begin_trigger4 = i - 15 - 48 - 5
        hc_trigger_data["4"]["begin"].append(begin_trigger4)
        hc_trigger_data["4"]["duration"].append(45)

    for i in hc_trigger_data["4"]["begin"]:
        end_time = i + 45
        hc_trigger_data["4"]["end"].append(end_time)

    # Set new annotations for the current file
    onsets = []
    durations = []
    descriptions = []

    # Accumulate trigger data into lists
    for description, data in hc_trigger_data.items():
        for onset, duration in zip(data["begin"], data["duration"]):
            onsets.append(onset)
            durations.append(duration)
            descriptions.append(description)

    # Create new annotations
    new_annotations = mne.Annotations(
    onsets,
    durations,
    descriptions,
    ch_names=None  # Set to None since annotations don't have associated channel names
    )
    
    raw_new=raw.set_annotations(new_annotations)
    
    #raw_new.plot(block=True)
    
    raw_trigger4 = raw_new.copy().crop(tmin=hc_trigger_data['4']['begin'][0], tmax=hc_trigger_data['4']['end'][0])
    raw_trigger5 = raw_new.copy().crop(tmin=hc_trigger_data['5']['begin'][0], tmax=hc_trigger_data['5']['end'][0])
    raw_trigger6 = raw_new.copy().crop(tmin=hc_trigger_data['6']['begin'][0], tmax=hc_trigger_data['6']['end'][0])
    raw_trigger7 = raw_new.copy().crop(tmin=hc_trigger_data['7']['begin'][0], tmax=hc_trigger_data['7']['end'][0])

    return (raw_trigger4, raw_trigger5, raw_trigger6, raw_trigger7)

def epochs_from_raw(raw,raw_trigger4, raw_trigger5, raw_trigger6, raw_trigger7):
    """
    Returns a mne.Epochs object created from the annotated mne.io.Raw object.

    """
    # Trigger 4 
    tstep = 3
    events = mne.make_fixed_length_events(raw, duration=tstep)

    interval = (-0.1,0)
    epochs_trigger4 = mne.Epochs(raw_trigger4, events, tmin=-0.1, tmax=tstep,
                    baseline=interval,
                    preload=True)
    epochs_trigger5 = mne.Epochs(raw_trigger5, events, tmin=-0.1, tmax=tstep,
                    baseline=interval,
                    preload=True)
    epochs_trigger6 = mne.Epochs(raw_trigger6, events, tmin=-0.1, tmax=tstep,
                    baseline=interval,
                    preload=True)
    epochs_trigger7 = mne.Epochs(raw_trigger7, events, tmin=-0.1, tmax=tstep,
                    baseline=interval,
                    preload=True)

    n_interpolates = np.array([1,4,32])
    consensus_percs = np.linspace(0,1.0,11)

    ar = AutoReject(n_interpolates, consensus_percs, thresh_method = "random_search", random_state = 42)

    # Trigger 4
    # Autoreject
    ar.fit(epochs_trigger4)
    epochs4_autor = ar.transform(epochs_trigger4)
    reject = autoreject.get_rejection_threshold(epochs4_autor, ch_types = "eeg")
    epochs4_autor.drop_bad(reject=reject)

    # Trigger 5
    # Autoreject
    ar.fit(epochs_trigger5)
    epochs5_autor = ar.transform(epochs_trigger5)
    reject = autoreject.get_rejection_threshold(epochs5_autor, ch_types = "eeg")
    epochs5_autor.drop_bad(reject=reject)

    # Trigger 6
    # Autoreject
    ar.fit(epochs_trigger6)
    epochs6_autor = ar.transform(epochs_trigger6)
    reject = autoreject.get_rejection_threshold(epochs6_autor, ch_types = "eeg")
    epochs6_autor.drop_bad(reject=reject)

    # Trigger 7  
    # Autoreject
    ar.fit(epochs_trigger4)
    epochs7_autor = ar.transform(epochs_trigger7)
    reject = autoreject.get_rejection_threshold(epochs7_autor, ch_types = "eeg")
    epochs7_autor.drop_bad(reject=reject)

    return (epochs4_autor, epochs5_autor, epochs6_autor, epochs7_autor)

"""

 -----------------------Time-Frequency Analysis----------------------------------------------------------------

"""

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

#assert test_xdf_files_reading()

mne.set_config("MNE_BROWSER_BACKEND", "qt")
paths = list()

# Read the data hc baseline fnirs
#path_hc_baseline_eeg = "H:\\Dokumenter\\data_processing\\data_eeg\\healthy_controls\\baseline\\try\\"
path_hc_baseline_eeg="/Users/adriana/Documents/GitHub/MasterThesis/try/"
folder_hc_baseline_eeg = os.fsencode(path_hc_baseline_eeg)

for file in os.listdir(folder_hc_baseline_eeg):
    filename = os.fsdecode(file)
    if filename.endswith(".xdf"):
        fname = path_hc_baseline_eeg + filename
        paths.append(fname)

for path in paths:
    print("\nNow working with file", path, "\n")

    # --------------Part 1: Read Raw data + Bad channel detection + Filter-------------------
    raw = get_raw_from_xdf(path)
    print(raw.first_samp)
    raw.load_data()
    raw = raw.copy()
    raw.set_eeg_reference(ref_channels="average")

    #raw.plot()

    raw.crop(tmin=170, tmax=raw.times[-1])
    raw.load_data()

    set_montage(raw)
    #raw.plot(title="Raw EEG", n_channels = len(raw.ch_names),show=True,block=True)
    #plt.show()

    # use pyprep to get rid of noisy channels and use interpolation
    noisy_channels = NoisyChannels(raw)
    noisy_channels.find_bad_by_SNR()
    noisy_channels.find_bad_by_deviation()
    noisy_channels.find_bad_by_hfnoise()
    print("Bad channels found by pyprep: ", noisy_channels.get_bads())
    raw.info["bads"]=noisy_channels.get_bads()
    #raw.plot(title="Noisy Channels Raw EEG", n_channels = len(raw.ch_names),show=True,block=True)
    #plt.show()
    
    # interpolate bad channels
    raw = raw.copy().pick(picks="eeg")
    raw_interp = raw.copy().interpolate_bads(reset_bads=False)
    # eeg_data.plot(block=True, butterfly=True,bad_color="r")
    # eeg_data_interp.plot(block=True, butterfly=True,bad_color="r")

    raw_filt=filter_eeg(raw_interp)
    # raw.plot(title="Raw EEG after preprocessing", n_channels = len(raw.ch_names),show=True,block=True)

    # --------------ICA artifact removal--------------------------------

    raw = repair_artifacts_ICA(raw, raw_filt)
    #raw.plot(block=True)

    # --------------Part 5: Characterization Raw data into N-back test events--------------------------

    triggers=characterization_trigger_data(raw)

    raw_trigger4 = triggers[0]
    raw_trigger5 = triggers[1]
    raw_trigger6 = triggers[2]
    raw_trigger7 = triggers[3]
 
    epochs = epochs_from_raw(raw, raw_trigger4, raw_trigger5, raw_trigger6, raw_trigger7)
