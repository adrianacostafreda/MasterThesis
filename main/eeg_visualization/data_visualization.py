import pandas as pd
import os
import basic.process_group_psd_data as process_psd
import basic.arrange_files as arrange
from basic.statistics import apply_stat_test
#from data_visualization.data_correlation import plot_correlation
from visualization.plot_psd import plot_boxplot_band

from IPython.display import display

os.chdir("H:\Dokumenter\GitHub\MasterThesis\.venv")

"""
VARIABLES TO CHANGE
psd_{xxx}_folder: A relative folder location of the regional/channels/asymmetry PSD files
exp_folder: A relative folder location of the comparable experiments as a list of strings
condition_code_list: A list of compared experiments' short codes
stat_test: The name of the statistical test to be used

We can change to BASELINE
psd_reg_folder = r"Results/baseline/Absolute PSD/regions/"
psd_ch_folder = r"Results/baseline/Absolute PSD/channels/"
psd_fooof_folder = r"Results/baseline/FOOOF/"

"""

psd_reg_folder = "H:\\Dokumenter\\data_processing\\Results\\n_back\\Absolute PSD\\regions\\"
psd_ch_folder = "H:\\Dokumenter\\data_processing\\Results\\n_back\\Absolute PSD\\channels\\"
psd_fooof_folder = "H:\\Dokumenter\\data_processing\\Results\\n_back\\FOOOF\\"
condition_legend = ['N-Back','Baseline']
stat_test = 'Wilcoxon'

exp_folder = ["0_back", "1_back", "2_back", "3_back"]
condition_codes = ['0-Back','1-Back', '2-Back', '3-Back']
condition_codes_comparisons = [['0-Back','1-Back', '2-Back', '3-Back']]

print('N-back')
[df_psd_reg_ec,df_psd_ch_ec,epochs_ec] = process_psd.read_group_psd_data(psd_reg_folder, psd_ch_folder,
                                                    exp_folder, non_responders=None, data_folder = "H:\\Dokumenter\\data_acquisition\\data_eeg\\clean\\n_back\\")
# export_group_psd_comparison(psd_reg_folder,psd_ch_folder,df_psd_reg_ec,df_psd_ch_ec,stat_test,
#                             condition_codes_comparisons_ec,verbose=True)


# FOOOF merge two groups together!!!
# Locate all PSD files (regions, channels and asymmetry) and save their information
dir_inprogress_fooof,b_names_fooof,condition_fooof = [None]*len(exp_folder),[None]*len(exp_folder),[None]*len(exp_folder)
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

df_psd_fooof_frontal = df_psd_fooof[df_psd_fooof['Region']=='frontal region']
df_psd_fooof_parietal = df_psd_fooof[df_psd_fooof['Region']=='parietal region']

df_psd_fooof_frontal = df_psd_fooof_frontal[['Subject','Exponent','Offset','Frequency band','Condition']]
df_psd_fooof_parietal = df_psd_fooof_parietal[['Subject','Exponent','Offset','Frequency band','Condition']]

plot_boxplot_band(df_psd_fooof_frontal, regions=['Exponent','Offset'],band='FOOOF',
                      condition_comp_list=[['0_Back','1_Back', '2_Back', '3_Back']],figsize=(3.75,4.5),yscale='linear',
                      condition_legend=condition_codes, fnt=['sans-serif',8,8],palette="light:#5A9",ylabel='',
                      legend=True,title=True,stat_test='t-test_ind',ast_loc='inside',verbose=False,export=False)

plot_boxplot_band(df_psd_fooof_parietal,regions=['Exponent','Offset'],band='FOOOF',
                      condition_comp_list=condition_codes_comparisons,figsize=(3.75,4.5),yscale='linear',
                      condition_legend=condition_codes, fnt=['sans-serif',8,8],palette="light:#5A9",ylabel='',
                      legend=True,title=True,stat_test='t-test_ind',ast_loc='inside',verbose=False,export=False)


for b_name in (df_psd_reg_ec['Frequency band'].unique()):
    plot_boxplot_band(df_psd_reg_ec,regions=['frontal region','parietal region'],band=b_name,
                      condition_comp_list=condition_codes_comparisons,figsize=(3.75,4.5),yscale='log',
                      condition_legend=condition_codes,fnt=['sans-serif',8,8],palette="light:#5A9",
                      legend=True,title=True,stat_test='t-test_ind',ast_loc='inside',verbose=True,export=False)
