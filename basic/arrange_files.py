import os
import pandas as pd

# ========== Functions ==========
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
            subject_names.append(os.path.join(file).removesuffix(filetype))

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


def array_to_df(subjectname, epochs, array_channels):
    """
    Convert channel-based array to Pandas dataframe with channels' and subjects' names. 

    Parameters
    ----------
    fname: the filename 
    epochs: Epochs-type (MNE-Python) EEG file
    array_channels: An array with values for each channel 

    Returns
    df_channels: A dataframe with values for each channel

    """
    df_channels = pd.DataFrame(array_channels).T

    df_channels.columns = epochs.info.ch_names
    df_channels['Subject'] = subjectname
    df_channels.set_index('Subject', inplace = True)

    return df_channels


def df_channels_to_regions(df_psd_band, brain_regions):
    """
    Average channels together based on the defined brain regions.

    Parameters
    ----------
    df_psd_band: A dataframe with PSD values for each channel per subject
    brain_regions: A dictionary of brain regions and EEG channels which they contain
    drop_cols: List of columns which are not channel PSD data

    Returns
    -------
    df_psd_reg_band: A dataframe with PSD values for each brain region per subject
    """

    df_psd_reg_band = pd.DataFrame()
    for region in brain_regions:
        df_temp = df_psd_band[brain_regions[region]].copy().mean(axis=1)
        df_psd_reg_band = pd.concat([df_psd_reg_band, df_temp], axis=1)
        
    df_psd_reg_band.columns = brain_regions.keys()
    df_psd_reg_band.index.name = 'Subject'

    return df_psd_reg_band

def create_results_folders(exp_folder, results_folder='Results', abs_psd=False,
                           rel_psd=False, fooof = False, fooof_2 = False):
    """
    Dummy way to try to pre-create folders for PSD results before exporting them

    Parameters
    ----------
    exp_folder: A string with a relative directory to experiment folder (e.g. 'Eyes Closed\Baseline')
    """
    if abs_psd == True:
        try:
            os.makedirs(os.path.join('{}/{}/Absolute PSD/channels'.format(results_folder, exp_folder)))
        except FileExistsError:
            pass
        try:
            os.makedirs(os.path.join('{}/{}/Absolute PSD/regions'.format(results_folder, exp_folder)))
        except FileExistsError:
            pass
    
    if rel_psd == True:
        try:
            os.makedirs(os.path.join('{}/{}/Relative PSD/channels'.format(results_folder, exp_folder)))
        except FileExistsError:
            pass
        try:
            os.makedirs(os.path.join('{}/{}/Relative PSD/regions'.format(results_folder, exp_folder)))
        except FileExistsError:
            pass
    
    if fooof== True:
        try:
            os.makedirs(os.path.join('{}/{}/FOOOF'.format(results_folder, exp_folder)))
        except FileExistsError:
            pass
    
    if fooof_2== True:
        try:
            os.makedirs(os.path.join('{}/{}/FOOOF_2'.format(results_folder, exp_folder)))
        except FileExistsError:
            pass
    
    try:
        os.makedirs(os.path.join('{}/{}'.format(results_folder, exp_folder)))
    except FileExistsError:
        pass

