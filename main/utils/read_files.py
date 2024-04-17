from tqdm import tqdm
import os
from mnelab.io.xdf import read_raw_xdf
import mne

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
