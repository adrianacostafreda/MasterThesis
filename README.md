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

`eeg_main`

This is an EEG pipeline for resting and n-back task EEG pre-processing and analyses used at the Department of Neurology at Rigshospitalet by Adriana Costafreda. 

### Preprocessing

`/eeg_main/preprocessing` - - importing raw resting state EEG (.xdf) files, re-referencing, removing bad channels, applying bandpass (e.g., 1-60 Hz) FIR filter, removing EOG noise with ICA, dividing signal into n-back task and baseline, performing epoch artefact rejection with Autoreject algorithm, exporting the cleaned EEG signals (.fif).

