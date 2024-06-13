import mne
import numpy as np
import matplotlib.pyplot as plt
import os
from mne_icalabel import label_components
import pyxdf

from mnelab.io.xdf import read_raw_xdf
from autoreject import AutoReject
from pyprep import NoisyChannels
from mne.preprocessing import ICA
from mne import Annotations


from basic.arrange_files import read_files


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

    channel_names = raw.info["ch_names"]

    print("These are the channels", channel_names)

    for i in raw.info["ch_names"]: 
        if i in  ["ACC_X","ACC_Y","ACC_Z","Trigger_0", "NIRStarTriggers_0", "NIRStarTriggers_1"]:
            # Drop channels which are not needed for the analysis
            raw.drop_channels(i)
    
    # Set the electrode positions
    channel_mapping = {"Fp1":"AFp1", "Fz": "AFF1h", "F3": "AF7", "F7": "AFF5h", "F9":"FT7", "FC5": "FC5", "FC1":"FC3", "C3":"FCC3h",
           "T7":"FFC1h", "CP5": "FCC1h", "CP1": "CCP3h", "Pz":"CCP1h", "P3":"CP1", "P7": "TP7", "P9":"CPP3h", "O1":"P1",
           "Oz":"AFp2", "O2":"AFF2h", "P10":"AF8", "P8":"AFF6h", "P4":"FT8", "CP2":"FC6", "CP6":"FC4", "T8":"FCC4h",
           "C4":"FFC2h", "Cz":"FCC2h", "FC2":"CCP4h", "FC6": "CCP2h", "F10":"CP2", "F8":"P2", "F4":"CPP4h", "Fp2":"TP8"}
    
    raw.rename_channels(channel_mapping)
    cap_montage = mne.channels.make_standard_montage("brainproducts-RNP-BA-128")

    # Use the preloaded montage
    raw.set_montage(cap_montage)
    raw.set_eeg_reference(ref_channels="average")

    raw.plot_sensors(show_names=True)
    plt.show()
    
    return raw

def get_raw_from_xdf_attila(xdf_file_path: str, ref_electrode: str = "") -> mne.io.Raw:
    """This function loads an XDF EEG file, and returns the corresponding mne.io.Raw object.

    Parameters
    ----------
    xdf_file_path : str
        Path to the XDF file.
    ref_electrode : str
        If not empty, a referential montage with that electrode is used, otherwise an average montage is used.
    """
    streams, _ = pyxdf.load_xdf(xdf_file_path)


    # Find where the EEG data is located within the data structure
    assert len(streams) == 2, (
        "Unexpected XDF data structure : expecting 2 streams, got " + str(len(streams))
    )
    if streams[1]["time_series"].shape[0] > streams[0]["time_series"].shape[0]:
        stream_index = 1
        stream_index_markers = 0
    else:
        stream_index = 0
        stream_index_markers = 1

    #stream_index = 0

    # Count EEG channels and find the reference channel's index
    channels_info = streams[stream_index]["info"]["desc"][0]["channels"][0]["channel"]
    eeg_channel_count = 0
    ref_channel = -1
    for index, e in enumerate(channels_info):
        if e["type"][0] == "EEG":
            eeg_channel_count += 1
        if e["label"][0] == ref_electrode:
            ref_channel = index

    # Extract channels' info
    data = streams[stream_index]["time_series"].T
    # It is assumed that the EEG channels are the first ones
    data = data[:eeg_channel_count]
    # micro V to V and preamp gain ???
    data[:] *= 1e-6  # / 2
    sfreq = float(streams[stream_index]["info"]["nominal_srate"][0])
    channel_names = [
        e["label"][0]
        + (
            (" - " + ref_electrode)
            if (e["label"][0] != ref_electrode) and ref_electrode != ""
            else ""
        )
        for e in channels_info[:eeg_channel_count]
    ]

    # Data format check
    assert eeg_channel_count > 0, "No EEG channels were found."
    if ref_electrode != "":
        assert ref_channel > -1, "The specified reference electrode was not found."
    for e in channel_names:
        assert e[0] in ["F", "C", "T", "P", "O"], "The channel names are unexpected."
    assert sfreq > 0.0, "The sampling frequency is not a positive number."

    # Create the mne.io.Raw object
    info = mne.create_info(channel_names, sfreq, ["eeg"] * eeg_channel_count)
    raw = mne.io.RawArray(data, info, verbose=False)

    # Event annotations
    #origin_time = streams[stream_index]["time_stamps"][0]
    #markers_time_stamps = [
    #    e - origin_time for e in streams[stream_index_markers]["time_stamps"]
    #]
    #markers_nb = len(markers_time_stamps)
    #markers = Annotations(
    #    onset=markers_time_stamps,
    #    duration=[10] * 3 + [25] * 5 + [25] * 5,
    #    description=["Audio"] * 3
    #    + ["Mental arithmetics moderate"] * 5
    #    + ["Mental arithmetics hard"] * 5,
    #    ch_names=[channel_names] * markers_nb,
    #)
    #raw.set_annotations(markers)

    # Set the reference montage
    if ref_electrode != "":
        raw = raw.set_eeg_reference(ref_channels=[ref_electrode], verbose=False)
    else:
        raw = raw.set_eeg_reference(verbose=False)  # Use the average montage

    # Set the electrode positions
    channel_mapping =  {"Fp1":"AFp1", "Fz": "AFF1h", "F3": "AF7", "F7": "AFF5h", "F9":"FT7", "FC5": "FC5", "FC1":"FC3", "C3":"FCC3h",
           "T7":"FFC1h", "CP5": "FCC1h", "CP1": "CCP3h", "Pz":"CCP1h", "P3":"CP1", "P7": "TP7", "P9":"CPP3h", "O1":"P1",
           "Oz":"AFp2", "O2":"AFF2h", "P10":"AF8", "P8":"AFF6h", "P4":"FT8", "CP2":"FC6", "CP6":"FC4", "T8":"FCC4h",
           "C4":"FFC2h", "Cz":"FCC2h", "FC2":"CCP4h", "FC6": "CCP2h", "F10":"CP2", "F8":"P2", "F4":"CPP4h", "Fp2":"TP8"}
    
    raw.rename_channels(channel_mapping)
    cap_montage = mne.channels.make_standard_montage("brainproducts-RNP-BA-128")
    raw.set_montage(cap_montage)

    raw.plot_sensors(show_names=True)
    plt.show()

    return raw

"""

 -----------------------Preprocessing----------------------------------------------------------------

"""

def filter_eeg(raw: mne.io.Raw) -> None:
    """
    Preprocess EEG data: filter (bandpass filter between 1 and 70 Hz) and a notch filter at 50 Hz
    """

    raw.filter(l_freq = 1, h_freq = 60, verbose = False)
    raw.notch_filter(50, verbose=False)
    return raw

def repair_artifacts_ICA(raw_filt):
    
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

    #ica.plot_sources(epochs_ica, block = True)
    #ica.plot_components()
    #plt.show()

    ic_labels = label_components(raw_filt, ica, method="iclabel")

    labels = ic_labels["labels"]
    
    exclude_idx = [
        idx for idx, label in enumerate(labels) if label in ["eye blink"]
    ]

    print(f"Excluding these ICA components: {exclude_idx}")

    # ica.apply() changes the Raw object in-place, so let's make a copy first:
    reconst_raw = raw_filt.copy()
    ica.apply(reconst_raw, exclude=exclude_idx)

    return reconst_raw


# Set default directory
os.chdir("H:\Dokumenter\GitHub\MasterThesis\.venv")
#os.chdir("/Users/adriana/Documents/GitHub/thesis/MasterThesis/")
mne.set_log_level('error')

# Folder where to get the raw EEG files
raw_folder = "H:\\Dokumenter\\data_acquisition\\data_eeg\\healthy_controls\\baseline\\raw\\"
#raw_folder = "/Users/adriana/Documents/DTU/thesis/data_acquisition/data_eeg/healthy_controls/"

# Folder where to export the clean epochs files
clean_raw_folder =  "H:\\Dokumenter\\data_acquisition\\data_eeg\\clean_eeg\\healthy_controls\\"
#clean_raw_folder = "/Users/adriana/Documents/DTU/thesis/data_acquisition/clean_eeg/healthy_controls/"

# Get directories of raw EEG files and set export directory for clean files
dir_inprogress = os.path.join(raw_folder)
export_dir = os.path.join(clean_raw_folder)

file_dirs, subject_names = read_files(dir_inprogress,'.xdf')

mne.set_config('MNE_BROWSER_BACKEND', 'matplotlib')

bad_epochs = list()

# Loop through all the subjects' directories (EEG files directories)
for i in range(len(file_dirs)):
    
    # --------------Read Raw data + Bad channel detection + Filter-------------------
    raw = get_raw_from_xdf(file_dirs[i])
    #raw = get_raw_from_xdf_attila(file_dirs[i])
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
    raw.interpolate_bads(reset_bads = True)

    # --------------Re-referencing EEG channels--------------------------------
    raw.load_data()
    raw = raw.set_eeg_reference(ref_channels="average")
    
    # --------------Filtering--------------------------------
    raw_filt = filter_eeg(raw)

    # --------------ICA artifact------------------------------
    reconst_raw = repair_artifacts_ICA(raw_filt)

    fname = '{}/{}_clean.fif'.format(export_dir, subject_names[i])

    reconst_raw.save(fname, overwrite = True)



