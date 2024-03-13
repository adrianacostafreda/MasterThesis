# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 15:37:46 2024

@author: AMAR0142
"""

from itertools import compress
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mne
import pyvistaqt
import ipywidgets
import mpl_toolkits
import os

from pyxdf import match_streaminfos, resolve_streams
from mnelab.io.xdf import read_raw_xdf, _resample_streams

# Read the data hc baseline fnirs
path_hc_baseline_eeg = "H:\\Dokumenter\\data_processing\\data_eeg\\healthy_controls\\baseline\\"
folder_hc_baseline_eeg = os.fsencode(path_hc_baseline_eeg)

hc_filenames_eeg_baseline = []
hc_files_eeg_baseline = []

# Dictionary to store trigger data
hc_trigger_data = {
    '4': {'begin': [], 'end': [], 'duration': []},
    '5': {'begin': [], 'end': [], 'duration': []},
    '6': {'begin': [], 'end': [], 'duration': []},
    '7': {'begin': [], 'end': [], 'duration': []}
}
start_times =[]

for file in os.listdir(folder_hc_baseline_eeg):
    filename = os.fsdecode(file)
    if filename.endswith(".xdf"):
        hc_filenames_eeg_baseline.append(filename)
        fname = path_hc_baseline_eeg + filename
        #streams = resolve_streams(fname)
        #stream_id = match_streaminfos(streams, [{"type":"EEG"}])
        print(fname)

        raw_hc_bl_eeg = read_raw_xdf(fname, stream_ids = [1,2], fs_new=500)
        raw_hc_bl_eeg.load_data()
        
        hc_files_eeg_baseline.append(raw_hc_bl_eeg)
        
        annot = raw_hc_bl_eeg.annotations
        print(raw_hc_bl_eeg)