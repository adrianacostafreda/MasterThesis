import pandas as pd
import os
import mne

import numpy as np
import matplotlib.pyplot as plt

import fooof

from fooof.bands import Bands
from fooof.utils import trim_spectrum
from fooof.analysis import get_band_peak_fm
from fooof.sim.gen import gen_power_spectrum
from fooof.plts.spectra import plot_spectra_shading

from IPython.display import display

# Set default directory
os.chdir("/Users/adriana/Documents/GitHub/thesis/MasterThesis/")
mne.set_log_level('error')

# Folder where to obtain the data
results_foldername = "/Users/adriana/Documents/DTU/thesis/data_processing/"

exp_folder = 'n_back/'
exp_test = 'FOOOF/'
group = "healthy_controls/"
location = 'central_region'

# Get directories of clean EEG files and set export directory
dir_inprogress = os.path.join(results_foldername, exp_folder, exp_test, group, location)

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

file_dirs, subject_names = read_files(dir_inprogress, '_fooof.xlsx', verbose=True)

# Define the conditions based on file names
conditions = {'0_back': [], '1_back': [], '2_back': [], '3_back': []}

for file in file_dirs:
    for condition in conditions.keys():
        if condition in file:
            conditions[condition].append(file)
            break

# Define the frequency bands of interest
band_labels = {'Delta': [1, 4], 'Theta': [4, 8], 'Alpha': [8, 13], 'Beta': [13, 30]}

# Initialize dictionaries to store periodic and aperiodic data for each condition
periodic_data = {key: {band: [] for band in band_labels} for key in conditions.keys()}
aperiodic_data = {key: {band: [] for band in band_labels} for key in conditions.keys()}

# Extract periodic and aperiodic data for each condition
for condition_key, condition_files in conditions.items():
    for file in condition_files:
        df = pd.read_excel(file)
        for band in band_labels:
            if band in file:
                # Extract periodic data

                periodic_data[condition_key][band].append(sum(df[f'{band} CF'].values.tolist())/len(df[f'{band} CF'].values.tolist()))
                periodic_data[condition_key][band].append(sum(df[f'{band} PW'].values.tolist())/len(df[f'{band} PW'].values.tolist()))
                periodic_data[condition_key][band].append(band_labels[band][1] - band_labels[band][0])
                # Extract aperiodic data
                aperiodic_data[condition_key][band].append(sum(df['Exponent'].values.tolist())/len(df['Exponent'].values.tolist()))
                aperiodic_data[condition_key][band].append(sum(df['Offset'].values.tolist())/len(df['Offset'].values.tolist()))

print(periodic_data)

# Store the simplified data in a dictionary
condition_data = {}

# Define condition keys
conditions = ['0_back', '1_back', '2_back', '3_back']

for condition in conditions:
    pe_key = f'pe_{condition}'
    ap_key = f'ap_{condition}'
    pe_data = [periodic_data[condition][band] for band in band_labels]
    ap_data = [aperiodic_data[condition][band] for band in band_labels]
    condition_data[pe_key] = pe_data
    condition_data[ap_key] = ap_data


# Define the frequency bands of interest
bands_fooof = Bands({'delta': [1,4],
               'theta': [4,8],
               'alpha': [8,13],
               'beta': [13,30]})

# Define plot settings 
t_settings = {'fontsize' : 24, 'fontweight': 'bold'}
shade_cols = ['#e8dc35', '#46b870', '#1882d9', '#a218d9']
labels = ['0-Back', '1-Back', '2-Back', '3-Back']

# General simulation settings
f_range = [1,30]
nlv = 0

freqs, back0_spectrum_bands = gen_power_spectrum(f_range, condition_data['ap_0_back'][0], condition_data['pe_0_back'], nlv)
freqs, back1_spectrum_bands = gen_power_spectrum(f_range, condition_data['ap_1_back'][0], condition_data['pe_1_back'], nlv)
freqs, back2_spectrum_bands = gen_power_spectrum(f_range, condition_data['ap_2_back'][0], condition_data['pe_2_back'], nlv)
freqs, back3_spectrum_bands = gen_power_spectrum(f_range, condition_data['ap_3_back'][0], condition_data['pe_3_back'], nlv)

plot_spectra_shading(freqs, [back0_spectrum_bands,back1_spectrum_bands,back2_spectrum_bands,back3_spectrum_bands],
                     log_powers=True, linewidth=3,
                     shades= bands_fooof.definitions, shade_colors=shade_cols,
                     labels=labels)
plt.xlim(f_range)
plt.title("Band-by-Band - " + group + location, t_settings)

# Initialize FOOOF objects

fm_bands_back0 = fooof.FOOOF(verbose=False)
fm_bands_back1 = fooof.FOOOF(verbose=False)
fm_bands_back2 = fooof.FOOOF(verbose=False)
fm_bands_back3 = fooof.FOOOF(verbose=False)

fm_bands_back0.fit(freqs, back0_spectrum_bands)
fm_bands_back1.fit(freqs, back1_spectrum_bands)
fm_bands_back2.fit(freqs, back2_spectrum_bands)
fm_bands_back3.fit(freqs, back3_spectrum_bands)

# Fit power spectra differences
plot_spectra_shading(freqs, [fm_bands_back0._spectrum_flat, fm_bands_back1._spectrum_flat,
                             fm_bands_back2._spectrum_flat, fm_bands_back3._spectrum_flat] ,
                    log_powers = False, linewidth=3, shades = bands_fooof.definitions, 
                    shade_colors=shade_cols,
                    labels=labels)
plt.xlim(f_range)
plt.title("Band-by-Band - Flattened - " + group + location, t_settings)
plt.show()

# Compare spectral parameters

def compare_exp(fm_bands_back0, fm_bands_back1, fm_bands_back2, fm_bands_back3):
    """ Compare exponent parameters """

    exp_0back = fm_bands_back0.get_params('aperiodic_params', 'exponent')
    exp_1back = fm_bands_back1.get_params('aperiodic_params', 'exponent')
    exp_2back = fm_bands_back2.get_params('aperiodic_params', 'exponent')
    exp_3back = fm_bands_back3.get_params('aperiodic_params', 'exponent')

    return (exp_0back-exp_1back, exp_0back-exp_2back, exp_0back-exp_3back)


def compare_peak_pw(fm_bands_back0, fm_bands_back1, fm_bands_back2, fm_bands_back3, band_def):
    """ Compare exponent parameters """

    pw_0back = get_band_peak_fm(fm_bands_back0, band_def)[1]
    pw_1back = get_band_peak_fm(fm_bands_back1, band_def)[1]
    pw_2back = get_band_peak_fm(fm_bands_back2, band_def)[1]
    pw_3back = get_band_peak_fm(fm_bands_back3, band_def)[1]
    

    return (pw_0back-pw_1back, pw_0back - pw_2back, pw_0back - pw_3back)

def compare_band_pw(fm_bands_back0, fm_bands_back1, fm_bands_back2, fm_bands_back3, band_def):
    """ Compare exponent parameters """

    pw_0back = np.mean(trim_spectrum(fm_bands_back0.freqs, fm_bands_back0.power_spectrum, band_def)[1])
    pw_1back = np.mean(trim_spectrum(fm_bands_back1.freqs, fm_bands_back1.power_spectrum, band_def)[1])
    pw_2back = np.mean(trim_spectrum(fm_bands_back2.freqs, fm_bands_back2.power_spectrum, band_def)[1])
    pw_3back = np.mean(trim_spectrum(fm_bands_back3.freqs, fm_bands_back3.power_spectrum, band_def)[1])

    return (pw_0back-pw_1back, pw_0back-pw_2back, pw_0back-pw_3back)

exp_template_1back = "The difference of aperiodic exponent between 0 back and 1 back is: \t {:1.2f}"
exp_template_2back = "The difference of aperiodic exponent between 0 back and 2 back is: \t {:1.2f}"
exp_template_3back = "The difference of aperiodic exponent between 0 back and 3 back is: \t {:1.2f}"
pw_template_1back = ("The difference of {:5} power between 0 back and 1 back is {: 1.2f}\t"
               "with peaks or {: 1.2f}\t with bands.")
pw_template_2back = ("The difference of {:5} power between 0 back and 2 back is {: 1.2f}\t"
               "with peaks or {: 1.2f}\t with bands.")
pw_template_3back = ("The difference of {:5} power between 0 back and 3 back is {: 1.2f}\t"
               "with peaks or {: 1.2f}\t with bands.")


print(exp_template_1back.format(compare_exp(fm_bands_back0,fm_bands_back1,fm_bands_back2,fm_bands_back3)[0]))
print(exp_template_2back.format(compare_exp(fm_bands_back0,fm_bands_back1,fm_bands_back2,fm_bands_back3)[1]))
print(exp_template_3back.format(compare_exp(fm_bands_back0,fm_bands_back1,fm_bands_back2,fm_bands_back3)[2]))

for label, definition in bands_fooof:
    print(pw_template_1back.format(label,
                                   compare_peak_pw(fm_bands_back0,fm_bands_back1,fm_bands_back2,fm_bands_back3,definition)[0],
                                   compare_band_pw(fm_bands_back0,fm_bands_back1,fm_bands_back2,fm_bands_back3,definition)[0]))
    print(pw_template_2back.format(label,
                                   compare_peak_pw(fm_bands_back0,fm_bands_back1,fm_bands_back2,fm_bands_back3,definition)[1],
                                   compare_band_pw(fm_bands_back0,fm_bands_back1,fm_bands_back2,fm_bands_back3,definition)[1]))
    print(pw_template_3back.format(label,
                                   compare_peak_pw(fm_bands_back0,fm_bands_back1,fm_bands_back2,fm_bands_back3,definition)[2],
                                   compare_band_pw(fm_bands_back0,fm_bands_back1,fm_bands_back2,fm_bands_back3,definition)[2]))