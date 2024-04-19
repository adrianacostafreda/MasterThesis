import mne
import numpy as np
import matplotlib.pyplot as plt
from mne.preprocessing import ICA
import os
from mnelab.io.xdf import read_raw_xdf
import autoreject
from autoreject import AutoReject
from pyprep import NoisyChannels
import mne_icalabel

"""

 -----------------------Reading EEG data----------------------------------------------------------------

"""

def get_raw_from_xdf(xdf_file_path: str):
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
     #raw_montage.plot_sensors(show_names = True)
     #plt.show()
    
    return raw_montage

"""

 -----------------------Preprocessing----------------------------------------------------------------

"""

def filter_eeg(raw: mne.io.Raw) -> None:
    """Preprocess EEG data: filter (bandpass filter between 0.1 and 30 Hz)
    
    Source : https://neuraldatascience.io/7-eeg/erp_filtering 

    """
    raw.filter(l_freq = 1, h_freq = 50, verbose = False)
    raw.notch_filter(np.arange(50, 250, 50), verbose=False)
    return

def repair_artifacts_ICA(raw, filt_raw):
    
    """
    Use Independent Component Analysis for artifact repair
    """

    # Break raw data into 1 s epochs
    tstep = 1.0
    events_ica = mne.make_fixed_length_events(raw, duration=tstep)
    epochs_ica = mne.Epochs(raw, events_ica,
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
    ica = ICA(n_components=ica_n_components,
                            random_state=random_state,
                            )
    ica.fit(epochs_ica[~reject_log.bad_epochs], decim=3)
    ica

    raw.load_data()
    ica.plot_sources(raw, show_scrollbars=False, show=True)

    ic_labels = mne_icalabel.label_components(filt_raw, ica, method="iclabel")
    print(ic_labels["labels"])
    
    #reconst_raw = raw.copy()
    #ica.apply(reconst_raw)

    #raw.plot(block=True)
    #reconst_raw.plot(block=True)

    return
        
"""
----------Define Epochs as Events for the N-Back Test-------------------------------

"""

def set_trigger4(raw):
    """Define an annotation for the Trigger 4"""

    annot = raw.annotations
    
    onset_trigg4 = annot.onset[8]-15-40
    trigger4 = mne.Annotations(onset_trigg4, 48, "4", orig_time = None)
    new_annotations = annot + trigger4
    raw.set_annotations(new_annotations)

    print(annot)

def epochs_from_raw(raw):
    """
    Returns a mne.Epochs object created from the annotated mne.io.Raw object.
    The length of the epochs is set to the duration of each trigger corresponding to the N-Back test

    Events are created around the events of interest which correspond to 4,5,6,7

    """

    events, _ = mne.events_from_annotations(raw) # Create events from existing annotations
    events = events[np.in1d(events[:,2], (4, 5, 6, 7)), : ]
    print("These are the events", events)
    
    event_id = {"Trigger 4": 4, "Trigger 5": 5, "Trigger 6": 6, "Trigger 7": 7}

    # Using this event array, we take continuous data and create epochs ranging from -0.5 seconds before to 60 seconds after each event for 5,6,7 and 48 seconds for 4 
    # In other words, an epoch comprises data from -0.5 to 60 seconds around 5, 6, 7 events. For event 4, the epoch will comprise data from -0.5 to 48 seconds. 
    # We will consider Fz, Cz, Pz channels corresponding to the mid frontal line
    tmin=-0.1
    tmax=60
    epochs = mne.Epochs(
        raw,
        events, 
        event_id,
        tmin,
        tmax,
        baseline=(-0.1,0),
        preload=True
    )

    return epochs

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
#path_hc_baseline_eeg = "H:\\Dokumenter\\data_processing\\data_eeg\\patients\\baseline\\try\\"
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
    raw = raw.copy()

    if 580 < raw.times[-1]:
        raw.crop(tmin=0, tmax=580)
    else:
        raw.crop(tmin=0, tmax=raw.times[-1])
    
    #raw.plot(title="Raw EEG with crop", n_channels = len(raw.ch_names),show=True,block=True)
    raw.load_data()

    set_montage(raw)
    raw.plot(title="Raw EEG", n_channels = len(raw.ch_names),show=True,block=True)
    plt.show()

    filt_raw = filter_eeg(raw)

    # use pyprep to get rid of noisy channels and use interpolation
    noisy_channels = NoisyChannels(filt_raw)
    
    noisy_channels.find_bad_by_deviation()
    noisy_channels.find_bad_by_hfnoise()
    print("Bad channels found by pyprep: ", noisy_channels.get_bads())
    filt_raw.info["bads"]=noisy_channels.get_bads()
    #raw.plot(title="Noisy Channels Raw EEG", n_channels = len(raw.ch_names),show=True,block=True)
    
    # interpolate bad channels
    raw = filt_raw.copy().pick(picks="eeg")
    raw_interp = filt_raw.copy().interpolate_bads(reset_bads=False)
    #raw.plot(title="Raw EEG without interpolated channels", block=True, butterfly=True,bad_color="r")
    #raw_interp.plot(title="Raw EEG data interpolating channels", block=True, butterfly=True,bad_color="r")

    raw_interp.set_eeg_reference(ref_channels="average")

    # --------------Part 2: ICA artifact removal--------------------------------

    raw = repair_artifacts_ICA(raw_interp,filt_raw)
    #raw.plot(title="Raw EEG after ICA", n_channels = len(raw.ch_names),show=True,block=True)

    # --------------Part 5: Characterization Raw data into N-back test events--------------------------

    set_trigger4(raw)
    epochs = epochs_from_raw(raw)
    
    reject = autoreject.get_rejection_threshold(epochs, ch_types = "eeg", cv=3)
    print("The rejection dictionary is %s" % reject)
    epochs.drop_bad(reject=reject)
    epochs.plot(title="Epoched EEG after epoch rejection", events = True, block=True)

    epochs.plot(title="Epoched EEG", events=True, block=True)
    epochs.average().plot()

    plot_spectrum(raw)

