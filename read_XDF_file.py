import os
import pathlib

import mne 

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from pyxdf import match_streaminfos, resolve_streams
from mnelab.io.xdf import read_raw_xdf

import pyprep
from pyprep.prep_pipeline import PrepPipeline

path = r"H:\Dokumenter\data_processing\data_eeg\healthy_controls\baseline\C1.xdf"
streams = resolve_streams(path)
stream_id = match_streaminfos(streams, [{"type":"EEG"}])[0]
raw = read_raw_xdf(path, stream_ids = [1,2],fs_new=500)
raw.load_data()