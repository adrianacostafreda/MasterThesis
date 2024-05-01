import mne
import numpy as np
import matplotlib.pyplot as plt
import os
from mnelab.io.xdf import read_raw_xdf
from autoreject import AutoReject, get_rejection_threshold
from pyprep import NoisyChannels
from mne.preprocessing import ICA
from mne_icalabel import label_components

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

    raw = read_raw_xdf(xdf_file_path, stream_ids = [1,2], fs_new=500)

    # Drop channels which are not needed for the analysis
    raw.drop_channels(["ACC_X","ACC_Y","ACC_Z","Trigger_0"])
    
    # Set the electrode positions
    channel_mapping = {"Fp1":"AFp1", "Fz": "AFF1h", "F3": "AF7", "F7": "AFF5h", "F9":"FT7", "FC5": "FC5", "FC1":"FC3", "C3":"FCC3h",
           "T7":"FFC1h", "CP5": "FCC1h", "CP1": "CCP3h", "Pz":"CCP1h", "P3":"CP1", "P7": "TP7", "P9":"CPP3h", "O1":"P1",
           "Oz":"AFp2", "O2":"AFF2h", "P10":"AF8", "P8":"AFF6h", "P4":"FT8", "CP2":"FC6", "CP6":"FC4", "T8":"FCC4h",
           "C4":"FFC2h", "Cz":"FCC2h", "FC2":"CCP4h", "FC6": "CCP2h", "F10":"CP2", "F8":"P2", "F4":"CPP4h", "Fp2":"TP8"}
    
    raw.rename_channels(channel_mapping)
    cap_montage = mne.channels.make_standard_montage("brainproducts-RNP-BA-128")

    # Use the preloaded montage
    raw.set_montage(cap_montage)
    
    return raw

"""

 -----------------------Preprocessing----------------------------------------------------------------

"""

def filter_eeg(raw: mne.io.Raw) -> None:
    """
    Preprocess EEG data: filter (bandpass filter between 1 and 70 Hz) and a notch filter at 50 Hz
    """

    raw.filter(l_freq = 1, h_freq = 70, verbose = False)
    raw.notch_filter(50, verbose=False)
    return raw

def repair_artifacts_ICA(raw, raw_filt):
    
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
    ica = ICA(n_components=ica_n_components,
                random_state=random_state,
                )
    ica.fit(epochs_ica[~reject_log.bad_epochs], decim=3)

    ica.plot_sources(epochs_ica, block = True)
    ica.plot_components()
    plt.show()

    ic_labels = label_components(raw_filt, ica, method="iclabel")

    labels = ic_labels["labels"]
    print(labels)
    
    exclude_idx = [
        idx for idx, label in enumerate(labels) if label in ["eye blink"]
    ]

    print(f"Excluding these ICA components: {exclude_idx}")

    # ica.apply() changes the Raw object in-place, so let's make a copy first:
    reconst_raw = raw.copy()
    ica.apply(reconst_raw, exclude=exclude_idx)

    return reconst_raw

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


mne.set_config("MNE_BROWSER_BACKEND", "qt")
paths = list()

# Read the data hc baseline fnirs
path_hc_baseline_eeg = "H:\\Dokumenter\\data_processing\\data_eeg\\healthy_controls\\baseline\\try\\"
#path_hc_baseline_eeg="/Users/adriana/Documents/GitHub/MasterThesis/try/"
folder_hc_baseline_eeg = os.fsencode(path_hc_baseline_eeg)

for file in os.listdir(folder_hc_baseline_eeg):
    filename = os.fsdecode(file)
    if filename.endswith(".xdf"):
        fname = path_hc_baseline_eeg + filename
   
        # --------------Part 1: Read Raw data + Bad channel detection + Filter-------------------
        raw = get_raw_from_xdf(fname)
        raw.load_data()
    
        # --------------Bad Channels--------------------------------

        # use pyprep to get rid of noisy channels and use interpolation
        noisy_channels = NoisyChannels(raw)
        noisy_channels.find_bad_by_deviation() # high/low overall amplitudes
        noisy_channels.find_bad_by_hfnoise() # high-frequenct noise
        bad_channels = noisy_channels.get_bads()
    
        print("Bad channels found by pyprep: ", noisy_channels.get_bads())
        raw.info["bads"] = bad_channels

        # interpolate bad channels
        raw.interpolate_bads(reset_bads=False)

        # --------------Re-referencing EEG channels--------------------------------
        raw.set_eeg_reference(ref_channels="average")
    
        # --------------Filtering--------------------------------
        raw_filt = filter_eeg(raw)

        # --------------ICA artifact------------------------------
        reconst_raw = repair_artifacts_ICA(raw, raw_filt)

        # --------------Epochs--------------------------------
        set_trigger4(reconst_raw)
        epochs = epochs_from_raw(reconst_raw).load_data()
        del reconst_raw

        # --------------Save as file--------------------------------
        file_name = filename + "_clean-epo.fif" # Same path, different file name
        epochs.save(file_name, overwrite=True)

        



    