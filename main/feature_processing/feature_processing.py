import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn import decomposition, preprocessing
from matplotlib.colors import ListedColormap

path_hc_baseline_eeg="/Users/adriana/Documents/GitHub/MasterThesis/"
#path_hc_baseline_eeg="H:\\Dokumenter\\Visual_Studio\\data_processing_eeg\\.venv\\"

# ---- Load ----
df = pd.read_csv(os.path.join(path_hc_baseline_eeg, "eeg_features.csv"))

baseline_df = df[df['baseline'] == "yes"]
trigger_df = df[df['baseline'] == "no"]


for df in [baseline_df, trigger_df]:

    for index, row in df.iterrows():

        id = row['patient_id']
        nback = row['n-back']
        