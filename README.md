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

`/eeg_main`###

This is an EEG pipeline for resting and n-back task EEG pre-processing and analyses used at the Department of Neurology at Rigshospitalet by Adriana Costafreda. 

### Preprocessing

`/eeg_main/preprocessing/preprocess.py` - importing raw resting state EEG (.xdf) files, re-referencing, removing bad channels, applying bandpass (e.g., 1-60 Hz) FIR filter, removing EOG noise with ICA, dividing signal into n-back task and baseline, performing epoch artefact rejection with Autoreject algorithm, exporting the cleaned EEG signals (.fif).

### Spectral analysis

`/eeg_main/spectrum/eeg_bp_classic.py` - It reads EEG clean files, calculates power spectral density (PSD) using the Welch method, and organizes the results into brain regions. Additionally, it computes the band power for specific frequency bands (Delta, Theta, Alpha, Beta) across EEG channels and brain regions. The code handles data visualization (e.g., topographic maps of band power) and saves the results as Excel files for further analysis. Results are organized into absolute and relative PSDs for channels and brain regions.

`/eeg_main/spectrum/eeg_aperiodic_specific_bp.py` - This Python code is designed for analyzing EEG data using MNE-Python and the FOOOF package. This code reads the EEG clean files, calculates the PSD, and fits the FOOOF model to estimate aperiodic and periodic components of the spectrum. It visualizes the results, including the power spectrum and model fit, while also calculating bandpower metrics for specific frequency bands, such as theta. This setup is useful for neuroscience research involving the analysis of brain activity patterns.

### Entropy Complexity Measures 

`/eeg_main/entropy_complexity/entropy_complexity_nback.py`- this code analyses EEG data, focusing on the Lempel-Ziv Complexity (LZC) and Multiscale Sample Entropy (MSE) metrics. The script reads clean EEG epochs, it calculates the LZC for each epoch of each channel, averaging the results to obtain a single value per subject. Similarly, it computes the MSE for each channel across all epochs, averaging these values to summarize the complexity of the EEG signals. Finally, the results, including total MSE and scale-specific MSE values, are stored in a DataFrame for further analysis.




