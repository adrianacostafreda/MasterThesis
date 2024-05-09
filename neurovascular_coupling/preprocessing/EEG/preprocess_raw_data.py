import mne
import numpy as np
import matplotlib.pyplot as plt
import os
from mnelab.io.xdf import read_raw_xdf
from autoreject import AutoReject
from pyprep import NoisyChannels
from mne.preprocessing import ICA
from mne_icalabel import label_components

#from arrange_files import read_files

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

    #raw.plot(block=True)
    #plt.show()
    
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
#os.chdir("H:\Dokumenter\GitHub\MasterThesis\.venv")
os.chdir("/Users/adriana/Documents/GitHub/thesis/MasterThesis/")
mne.set_log_level('error')

# Folder where to get the raw EEG files
#raw_folder = "H:\\Dokumenter\\data_acquisition\\data_eeg\\healthy_controls\\baseline\\raw\\try\\"
raw_folder = "/Users/adriana/Documents/DTU/thesis/data_acquisition/data_eeg/healthy_controls/"

# Folder where to export the clean epochs files
#clean_folder =  "H:\\Dokumenter\\data_acquisition\\data_eeg\\clean\\healthy_controls\\"
clean_raw_folder = "/Users/adriana/Documents/DTU/thesis/data_acquisition/clean_eeg/healthy_controls/"

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


