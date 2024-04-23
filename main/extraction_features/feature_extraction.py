import pandas as pd
import os
import basic.process_group_psd_data as process_psd
import basic.arrange_files as arrange
from basic.statistics import apply_stat_test


from IPython.display import display

os.chdir("H:\Dokumenter\GitHub\MasterThesis\.venv")

psd_reg_folder = "H:\\Dokumenter\\data_processing\\Results\\n_back\\Relative PSD\\regions\\"
psd_ch_folder = "H:\\Dokumenter\\data_processing\\Results\\n_back\\Relative PSD\\channels\\"

# FOOOF band is only Theta
psd_fooof_folder = "H:\\Dokumenter\\data_processing\\Results\\n_back\\FOOOF\\"

condition_legend = ['N-Back','Baseline']
stat_test = 'Wilcoxon'

exp_folder = ["0_back", "1_back", "2_back", "3_back"]
condition_codes = ['0-Back','1-Back', '2-Back', '3-Back']
condition_codes_comparisons = [['0-Back','1-Back', '2-Back', '3-Back']]

print('N-back')
[df_psd_reg , df_psd_ch , epochs_ec] = process_psd.read_group_psd_data(psd_reg_folder, psd_ch_folder,
                                                    exp_folder, non_responders=None, data_folder = "H:\\Dokumenter\\data_acquisition\\data_eeg\\clean\\n_back\\")

df_psd_ch['Subject'] = df_psd_ch['Subject'].str.replace("_0back","").str.replace("_1back","").str.replace("_2back","").str.replace("_3back","")
df_psd_reg['Subject'] = df_psd_reg['Subject'].str.replace("_0back","").str.replace("_1back","").str.replace("_2back","").str.replace("_3back","")

# The FOOOF is done for the theta band 

# Locate all PSD files (regions, channels) and save their information
dir_inprogress_fooof, b_names_fooof, condition_fooof = [None]*len(exp_folder),[None]*len(exp_folder),[None]*len(exp_folder)
for i in range(len(exp_folder)):
    [dir_inprogress_fooof[i],b_names_fooof[i],condition_fooof[i]] = arrange.read_excel_psd(exp_folder[i],psd_fooof_folder,verbose=False)

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

# Assuming df_psd_reg is your DataFrame
for index, row in df_psd_reg.iterrows():
    features_theta = []
    features_delta = []

    # Check if the value in the "Frequency band" column is "Theta" or "Delta"
    if row["Frequency band"] == "Theta":
        features_theta.append(row["Subject"])
        features_theta.append(row["Condition"])
        features_theta.append(row["frontal region"])
        features_theta.append(row["central region"])
    
    elif row["Frequency band"] == "Delta":
        features_delta.append(row["left temporal"])
        features_delta.append(row["right temporal"])
    
    
    # Append the features list to the features_list
    features_list_theta.append(features_theta)
    features_list_delta.append(features_delta)

features_list_theta = [lst for lst in features_list_theta if lst]
features_list_delta = [lst for lst in features_list_delta if lst]

df_theta = pd.DataFrame(features_list_theta, columns =['patient_id', 'n-back','relative power frontal region', 'relative power central region'])
df_delta = pd.DataFrame(features_list_delta, columns =['relative power left temporal', 'relative power right temporal'])
df_psd_fooof_frontal = df_psd_fooof_frontal[['Exponent','Offset']]

# Reset index of df1 and df2
df_theta.reset_index(drop=True, inplace=True)
df_delta.reset_index(drop=True, inplace=True)
df_psd_fooof_frontal.reset_index(drop=True, inplace=True)

concatenated_df = pd.concat([df_theta, df_delta, df_psd_fooof_frontal], axis = 1)

print(concatenated_df)
