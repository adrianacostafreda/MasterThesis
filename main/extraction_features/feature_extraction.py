import pandas as pd
import os
#import basic.process_group_psd_data as process_psd
#import basic.arrange_files as arrange
import mne

from IPython.display import display

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

def read_excel_psd(exp_folder, psd_folder, condition_strsplit='_psd_', verbose=True):
    """
    Get all PSD file directories and corresponding bands and experiment conditions.

    Parameters
    ----------
    exp_folder: A string with a relative directory to experiment folder (e.g. 'Eyes Closed\Baseline')
    psd_folder: A string with a relative directory to the results folder (e.g. 'Results\PSD\regions')

    Returns
    -------
    dir_inprogress: A string with directory to look for files
    b_names: A list of strings for frequency bands of the files
    condition: A list of strings for experiment conditions of the files
    """
    dir_inprogress = os.path.join(psd_folder,exp_folder)
    _, b_names = read_files(dir_inprogress,".xlsx",verbose=verbose)

    condition = [None]*len(b_names)
    for i in range(len(b_names)):
        condition[i] = b_names[i].split(condition_strsplit, 1)
    
    return [dir_inprogress, b_names, condition]

def read_group_psd_data(psd_reg_folder, psd_ch_folder, exp_folder, non_responders=None, data_folder="/Users/adriana/Documents/DTU/thesis/data_acquisition/healthy_controls/n_back/"):
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

    epochs = mne.read_epochs(fname='{}/{}_clean-epo.fif'.format(dir_inprogress_epo, subject_names[0]),verbose=False)

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

current_directory = "/Users/adriana/Documents/GitHub/thesis/MasterThesis/"

os.chdir("/Users/adriana/Documents/GitHub/thesis/MasterThesis/")

psd_reg_folder = "/Users/adriana/Documents/DTU/thesis/data_processing/Results/healthy_controls/n_back/Relative PSD/regions/"
psd_ch_folder =  "/Users/adriana/Documents/DTU/thesis/data_processing/Results/healthy_controls/n_back/Relative PSD/channels/"

# FOOOF band is only Theta
psd_fooof_folder =  "/Users/adriana/Documents/DTU/thesis/data_processing/Results/healthy_controls/n_back/FOOOF/"

condition_legend = ['healthy_controls','patients']
stat_test = 'Wilcoxon'

exp_folder = ["0_back", "1_back", "2_back", "3_back"]
condition_codes = ['0-Back','1-Back', '2-Back', '3-Back']
condition_codes_comparisons = [['0-Back','1-Back', '2-Back', '3-Back']]

print('N-back')
[df_psd_reg , df_psd_ch , epochs_ec] = read_group_psd_data(psd_reg_folder, psd_ch_folder,
                                                    exp_folder, non_responders=None, data_folder = "/Users/adriana/Documents/DTU/thesis/data_acquisition/healthy_controls/n_back/")

df_psd_ch['Subject'] = df_psd_ch['Subject'].str.replace("_0back","").str.replace("_1back","").str.replace("_2back","").str.replace("_3back","")
df_psd_reg['Subject'] = df_psd_reg['Subject'].str.replace("_0back","").str.replace("_1back","").str.replace("_2back","").str.replace("_3back","")

# The FOOOF is done for the theta band 

# Locate all PSD files (regions, channels) and save their information
dir_inprogress_fooof, b_names_fooof, condition_fooof = [None]*len(exp_folder),[None]*len(exp_folder),[None]*len(exp_folder)
for i in range(len(exp_folder)):
    [dir_inprogress_fooof[i],b_names_fooof[i],condition_fooof[i]] = read_excel_psd(exp_folder[i],psd_fooof_folder,verbose=False)

# Read all REGIONAL spectral data and save to dataframe
df_psd_fooof = pd.DataFrame()
for i in range(len(b_names_fooof[0])):
    for n_exps in range(len(b_names_fooof)):
        globals()[b_names_fooof[n_exps][i]] = pd.read_excel('{}/{}.xlsx'\
                                            .format(dir_inprogress_fooof[n_exps], b_names_fooof[n_exps][i]))\
        .assign(**{'Frequency band':'FOOOF','Condition': condition_fooof[n_exps][i][0]})
        df_psd_fooof = pd.concat([df_psd_fooof,globals()[b_names_fooof[n_exps][i]]])

df_psd_fooof['Condition'] = df_psd_fooof['Condition'].str.replace("_fooof","")

df_psd_fooof[['Condition', 'Region']] = df_psd_fooof['Condition'].str.rsplit('_', n=1, expand=True)
df_psd_fooof = df_psd_fooof.rename(columns={'Unnamed: 0': 'Subject'})
df_psd_fooof['Subject'] = df_psd_fooof['Subject'].str.replace("_0back","").str.replace("_1back","").str.replace("_2back","").str.replace("_3back","")

df_psd_fooof_frontal = df_psd_fooof[df_psd_fooof['Region']=='frontal region']

df_psd_fooof_frontal = df_psd_fooof_frontal[['Subject','Exponent','Offset','Frequency band','Condition']]

# Define features_list outside the loop
features_list_theta = []
features_list_delta = []
features_list_alpha = []

# Assuming df_psd_reg is your DataFrame
for index, row in df_psd_reg.iterrows():
    features_theta = []
    features_delta = []
    features_alpha = []

    # Check if the value in the "Frequency band" column is "Theta" or "Delta"
    if row["Frequency band"] == "Theta":
        features_theta.append(row["Subject"])
        features_theta.append(row["Condition"])
        features_theta.append(row["frontal region"])
        #features_theta.append(row["central region"])
        #features_theta.append(row['left temporal'])
        #features_theta.append(row['right temporal'])
        #features_theta.append(row['parietal region'])
        #features_theta.append(row['occipital region'])
    
    elif row["Frequency band"] == "Alpha":
        features_alpha.append(row["frontal region"])
        features_alpha.append(row["central region"])
        features_alpha.append(row['left temporal'])
        features_alpha.append(row['right temporal'])
        features_alpha.append(row['parietal region'])
        features_alpha.append(row['occipital region'])
    
    elif row["Frequency band"] == "Delta":
        #features_delta.append(row["left temporal"])
        features_delta.append(row["right temporal"])
    
    
    # Append the features list to the features_list
    features_list_theta.append(features_theta)
    features_list_delta.append(features_delta)
    features_list_alpha.append(features_alpha)

features_list_theta = [lst for lst in features_list_theta if lst]
features_list_delta = [lst for lst in features_list_delta if lst]
features_list_alpha = [lst for lst in features_list_alpha if lst]

df_theta = pd.DataFrame(features_list_theta, columns =['patient_id', 'n_back', 'theta frontal region'])
#df_alpha = pd.DataFrame(features_list_alpha, columns =['alpha frontal region', 'alpha central region', 'alpha left temporal', 'alpha right temporal', 'alpha parietal region', 'alpha occipital region'])
#df_delta = pd.DataFrame(features_list_delta, columns =['delta right temporal'])
df_psd_fooof_frontal = df_psd_fooof_frontal[['Exponent']]

# Reset index of df1 and df2
df_theta.reset_index(drop=True, inplace=True)
#df_delta.reset_index(drop=True, inplace=True)
#df_alpha.reset_index(drop=True, inplace=True)
df_psd_fooof_frontal.reset_index(drop=True, inplace=True)

concatenated_df = pd.concat([df_theta, df_psd_fooof_frontal], axis = 1)

print(concatenated_df)

concatenated_df.to_csv(os.path.join(current_directory, "eeg_features.csv"), index = False)
