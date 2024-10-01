# Neurovascular Coupling measured by fNIRS-EEG as a biomarker for predicting cognitive functioning after critical illness

This repository contains the code developed for my master's thesis, which explores Neurovascular Coupling (NVC) in critically ill patients using Electroencephalography (EEG) 
and Functional Near-Infrared Spectroscopy (fNIRS). The project involves advanced signal processing techniques and statistical analysis to assess cognitive function by examining 
brain activity and blood flow responses during cognitive tasks.

The Python-based code implements signal processing tools for analyzing EEG signals (such as Theta power in the frontal midline) and fNIRS signals (specifically Oxygenated Hemoglobin 
(HbO) concentration in the frontal cortex). These signals were recorded during an n-back task paradigm, a standard method for evaluating working memory.

## Key features of this repository include:

- **Preprocessing pipelines** for EEG and fNIRS signals, including filtering and artifact removal.
- **Feature extraction techniques** to analyze power spectral densities, focusing on Theta band analysis.
- **Hemodynamic response analysis** using fNIRS data to capture changes in HbO concentration under cognitive load.
- **Statistical methods** to compare neural and hemodynamic responses between healthy individuals and critically ill patients across different task complexities.

The overall aim is to identify alterations in NVC as a potential biomarker for cognitive impairment in critically ill patients. Despite no statistically significant results,
the study provided valuable insights into the relationship between brain activity, blood flow, and cognitive load.

## EEG pipeline in Python

`/eeg_main`

This is an EEG pipeline for resting and n-back task EEG pre-processing and analyses used at the Department of Neurology at Rigshospitalet by Adriana Costafreda. 

### Preprocessing

`/eeg_main/preprocessing/preprocess.py` - importing raw resting state EEG (.xdf) files, re-referencing, removing bad channels, applying bandpass (e.g., 1-60 Hz) FIR filter, removing EOG noise with ICA, dividing signal into n-back task and baseline, performing epoch artefact rejection with Autoreject algorithm, exporting the cleaned EEG signals (.fif).

### Spectral analysis

`/eeg_main/spectrum/eeg_bp_classic.py` - It reads EEG clean files, calculates power spectral density (PSD) using the Welch method, and organizes the results into brain regions. Additionally, it computes the band power for specific frequency bands (Delta, Theta, Alpha, Beta) across EEG channels and brain regions. The code handles data visualization (e.g., topographic maps of band power) and saves the results as Excel files for further analysis. Results are organized into absolute and relative PSDs for channels and brain regions.

`/eeg_main/spectrum/eeg_aperiodic_specific_bp.py` - This Python code is designed for analyzing EEG data using MNE-Python and the FOOOF package. This code reads the EEG clean files, calculates the PSD, and fits the FOOOF model to estimate the aperiodic and periodic components of the spectrum. It visualizes the results, including the power spectrum and model fit, while also calculating bandpower metrics for specific frequency bands, such as theta. This setup is useful for neuroscience research involving the analysis of brain activity patterns.

### Entropy Complexity Measures 

`/eeg_main/entropy_complexity/entropy_complexity_nback.py`- this code analyses EEG data, focusing on the Lempel-Ziv Complexity (LZC) and Multiscale Sample Entropy (MSE) metrics. The script reads clean EEG epochs, it calculates the LZC for each channel epoch, averaging the results to obtain a single value per subject. Similarly, it computes the MSE for each channel across all epochs, averaging these values to summarize the complexity of the EEG signals. Finally, the results, including total MSE and scale-specific MSE values, are stored in a DataFrame for further analysis.

### EEG Feature Extraction

`/eeg_main/feature_extraction/feature_extraction_no_aperiodic_features.py` - This Python script processes EEG data to analyze the relative power spectral density (PSD) of theta waves in healthy controls across various n-back tasks. The script reads pre-processed PSD data, filtering it to include only the theta frequency band. The script separates the data into different DataFrames based on the n-back conditions (0-back to 3-back) and initializes empty NumPy arrays to store power data for each condition. It iterates through the subjects for each condition, extracting and reshaping their power data into the respective arrays. Finally, it saves the processed NumPy arrays for each n-back condition to a specified directory for further analysis.

`/eeg_main/feature_extraction/feature_extraction_with_fooof.py` - This code also processes EEG data to analyze Power Spectral Density (PSD) for various experimental conditions (n-back). It includes the FOOOF aperiodic components. The script extracts specific features related to different frequency bands (Theta, Delta, Alpha), particularly focusing on the theta band for FOOOF analysis. Finally, it combines the processed data, resets the indices for consistency, and exports the results as a CSV file named eeg_features.csv for further analysis.

`/eeg_main/feature_extraction/k_means.py` - This code implements k-means clustering using both a custom approach and the KMeans class from sklearn. The script loads EEG data from .npy files and evaluates the optimal number of clusters by running the k-means function multiple times and calculating average inter-cluster distances to identify the "elbow point." Results are visualized through plots showing the average inter-cluster distance and clustered data in a 2D space defined by the first two principal components. Additionally, the KMeans class is utilized to validate clustering results and generate an elbow curve for optimal cluster selection. This was done to evaluate whether the features could distinguish four groups corresponding to the n-back task. 

### Feature Processing 

`/eeg_main/feature_processing/` - This folder contains several scripts focused on developing a multi-class linear classifier model for analyzing EEG (electroencephalogram) data. This project aimed to preprocess and extract relevant features from the EEG signals to build a classifier that could effectively distinguish between the n-back tasks using the theta power in the mid-frontal cortex. Although the development of the multi-class linear classifier was a promising initial approach, it ultimately served as a preliminary test; the research trajectory evolved toward a different direction. 


## fNIRS pipeline in Python

`\fnirs_main`

### Preprocessing

`\fnirs_main\Hemo.py` - This file defines the class, which processes functional near-infrared spectroscopy (fNIRS) data from .snirf files. Upon initialization, it loads the data and performs optional preprocessing steps, including converting raw signals to optical density, rejecting and interpolating bad channels, and applying short channel regression to enhance the signal quality. The class provides methods to visualize the hemoglobin concentration data, remove short channels, and retrieve the processed data. Additionally, it implements techniques for correcting motion artifacts and filtering the data to isolate relevant frequency components. This class serves as a comprehensive tool for analyzing fNIRS data, streamlining the preprocessing workflow for further analysis or visualization.

`\fnirs_main\mne_to_numpy.py` - processes fNIRS data files, utilizing the HemoData class to load and preprocess the hemoglobin signals. It iterates over the data files in a specified directory, standardizing the hemoglobin concentration data for each subject. The script constructs a multi-dimensional NumPy array to store the preprocessed data. Additionally, it saves the processed data arrays to .npy files for subsequent analysis, providing separate datasets for each event (nback task) and a combined dataset for all events.













