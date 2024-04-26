# ========== Packages ==========
import mne
import os
import pandas as pd
from basic.arrange_files import read_excel_psd, read_files
from basic.statistics import apply_stat_test

# ========== Functions ==========
def read_group_psd_data(psd_reg_folder, psd_ch_folder, exp_folder, non_responders=None, data_folder="H:\\Dokumenter\\data_acquisition\\data_eeg\\clean\\n_back\\"):
    """
    Read and organise all the PSD data (before visualisation) together for one experiment state.

    Parameters
    ----------
    psd_reg_folder: A string with relative path to regional PSD values' files
    psd_ch_folder: A string with relative path to channels' PSD values' files
    exp_folder: A string with a relative directory to experiment folder (e.g. 'Eyes Closed\Baseline')
    non_responders (optional): A string for not including some of the subjects, for example for removing non-responders

    Returns
    -------
    df_psd_reg: A dataframe with PSD values for each region per subject
    df_psd_ch: A dataframe with PSD values for each channel per subject
    epochs: An epochs file for that experiment from the first subject
    """
    
    # Locate all PSD files (regions, channels and asymmetry) and save their information
    dir_inprogress_reg,b_names_reg,condition_reg = [None]*len(exp_folder),[None]*len(exp_folder),[None]*len(exp_folder)
    dir_inprogress_ch,b_names_ch,condition_ch = [None]*len(exp_folder),[None]*len(exp_folder),[None]*len(exp_folder)
    for i in range(len(exp_folder)):
        [dir_inprogress_reg[i],b_names_reg[i],condition_reg[i]] = read_excel_psd(exp_folder[i], psd_reg_folder, verbose=False)
        [dir_inprogress_ch[i],b_names_ch[i],condition_ch[i]] = read_excel_psd(exp_folder[i], psd_ch_folder, verbose=False)

    # Get one epochs file for later topographical plots' electrode placement information
    dir_inprogress_epo = os.path.join(r"{}".format(data_folder), exp_folder[0])
    _, subject_names = read_files(dir_inprogress_epo,"_clean-epo.fif",verbose=False)

    epochs = mne.read_epochs(fname='{}/{}_clean-epo.fif'.format(dir_inprogress_epo,subject_names[0]),verbose=False)

    # Read all REGIONAL spectral data and save to dataframe
    df_psd_reg = pd.DataFrame()
    for i in range(len(b_names_reg[0])):
        for n_exps in range(len(b_names_reg)):
            globals()[b_names_reg[n_exps][i]] = pd.read_excel('{}/{}.xlsx'\
                                                .format(dir_inprogress_reg[n_exps],b_names_reg[n_exps][i]))\
            .assign(**{'Frequency band': condition_reg[n_exps][i][1],'Condition': condition_reg[n_exps][i][0]})
            df_psd_reg = pd.concat([df_psd_reg,globals()[b_names_reg[n_exps][i]]])

    # Read all CHANNELS spectral data and save to dataframe
    df_psd_ch = pd.DataFrame()
    for i in range(len(b_names_ch[0])):
        for n_exps in range(len(b_names_ch)):
            globals()[b_names_ch[n_exps][i]] = pd.read_excel('{}/{}.xlsx'
                                                            .format(dir_inprogress_ch[n_exps],b_names_ch[n_exps][i]))\
                .assign(**{'Frequency band': condition_ch[n_exps][i][1],'Condition': condition_ch[n_exps][i][0]})
            df_psd_ch = pd.concat([df_psd_ch,globals()[b_names_ch[n_exps][i]]])

    # Option to remove some participants from further analysis (ex. removing non-responders of treatment)
    if non_responders != None:
        df_psd_reg = df_psd_reg[df_psd_reg['Subject'].str.contains(non_responders) == False]
        df_psd_ch = df_psd_ch[df_psd_ch['Subject'].str.contains(non_responders) == False]
    
    return [df_psd_reg, df_psd_ch, epochs]