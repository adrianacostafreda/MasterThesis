import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------------0 back--------------------------------------------------------------- 
path_hbo_0back = "H:\\Dokumenter\\data_processing\\Results_fNIRS\\features\\features_hbo_0Back.npy"
path_hbr_0back = "H:\\Dokumenter\\data_processing\\Results_fNIRS\\features\\features_hbr_0Back.npy"
path_eeg_0back = "H:\\Dokumenter\\data_processing\\Results_EEG\\features\\features_0Back.npy"

features_hbo_0back = np.load(path_hbo_0back)
features_hbr_0back = np.load(path_hbr_0back)
features_eeg_0back = np.load(path_eeg_0back)

# We take the mean of the HBO and only 22 patients to match dimensions with the EEG features
mean_hbo_0back = features_hbo_0back[0:22, :, 0]
mean_hbo_0back.reshape((22, 8, 1))

std_hbo_0back = features_hbo_0back[0:22, :, 1]
std_hbo_0back.reshape((22, 8, 1))

slope_hbo_0back = features_hbo_0back[0:22, :, 2]
slope_hbo_0back.reshape((22, 8, 1))

# We take the mean of the HBR and only 22 patients to match dimensions with the EEG features
mean_hbr_0back = features_hbr_0back[0:22, :, 0:1]
mean_hbr_0back.reshape((22, 8, 1))

std_hbr_0back = features_hbr_0back[0:22, :, 1:2]
std_hbr_0back.reshape((22, 8, 1))

slope_hbr_0back = features_hbr_0back[0:22, :, 2:3]
slope_hbr_0back.reshape((22, 8, 1))

# Compute the mean power across the channels for each subject
subjects_mean_hbo_0back = np.mean(mean_hbo_0back, axis=1).flatten()  # shape: (22, 1)
subjects_std_hbo_0back = np.mean(std_hbo_0back, axis=1).flatten()  # shape: (22, 1)
subjects_slope_hbo_0back = np.mean(slope_hbo_0back, axis=1).flatten()  # shape: (22, 1)
subjects_mean_hbr_0back = np.mean(mean_hbr_0back, axis=1).flatten()  # shape: (22, 1)
subjects_std_hbr_0back = np.mean(std_hbr_0back, axis=1).flatten()  # shape: (22, 1)
subjects_slope_hbr_0back = np.mean(slope_hbr_0back, axis=1).flatten()  # shape: (22, 1)
subjects_mean_eeg_0back = np.mean(features_eeg_0back, axis=1).flatten() # shape: (22,1)

# Compute the correlation matrix
correlation_matrix_0back = np.corrcoef(subjects_mean_hbo_0back, subjects_mean_eeg_0back, rowvar=False)  # rowvar = False means that each column represents a variable (subject)

# Print the correlation matrix
print("Correlation Matrix 0Back:")
print(correlation_matrix_0back)

# Create a df 

array_0Back = np.repeat(0, 22)

data_0Back = {
    "Mean Hbo": subjects_mean_hbo_0back,
    "Std Hbo": subjects_std_hbo_0back,
    "Slope Hbo": subjects_slope_hbo_0back,
    "Mean Hbr": subjects_mean_hbr_0back,
    "Std Hbr": subjects_std_hbr_0back,
    "Slope Hbr": subjects_slope_hbr_0back,
    "Theta Power Frontal Mid Line": subjects_mean_eeg_0back,
    "N_back" : array_0Back
}

# Create a DataFrame from the dictionary
df_0Back = pd.DataFrame(data_0Back)



# ----------------------------------------1 back--------------------------------------------------------------- 
path_hbo_1back = "H:\\Dokumenter\\data_processing\\Results_fNIRS\\features\\features_hbo_1Back.npy"
path_hbr_1back = "H:\\Dokumenter\\data_processing\\Results_fNIRS\\features\\features_hbr_1Back.npy"
path_eeg_1back = "H:\\Dokumenter\\data_processing\\Results_EEG\\features\\features_1Back.npy"

features_hbo_1back = np.load(path_hbo_1back)
features_hbr_1back = np.load(path_hbr_1back)
features_eeg_1back = np.load(path_eeg_1back)

# We take the mean of the HBO and only 22 patients to match dimensions with the EEG features
mean_hbo_1back = features_hbo_1back[0:22, :, 0]
mean_hbo_1back.reshape((22, 8, 1))

std_hbo_1back = features_hbo_1back[0:22, :, 1]
std_hbo_1back.reshape((22, 8, 1))

slope_hbo_1back = features_hbo_1back[0:22, :, 2]
slope_hbo_1back.reshape((22, 8, 1))

# We take the mean of the HBR and only 22 patients to match dimensions with the EEG features
mean_hbr_1back = features_hbr_1back[0:22, :, 0:1]
mean_hbr_1back.reshape((22, 8, 1))

std_hbr_1back = features_hbr_1back[0:22, :, 1:2]
std_hbr_1back.reshape((22, 8, 1))

slope_hbr_1back = features_hbr_1back[0:22, :, 2:3]
slope_hbr_1back.reshape((22, 8, 1))

# Compute the mean power across the channels for each subject
subjects_mean_hbo_1back = np.mean(mean_hbo_1back, axis=1).flatten()  # shape: (22, 1)
subjects_std_hbo_1back = np.mean(std_hbo_1back, axis=1).flatten()  # shape: (22, 1)
subjects_slope_hbo_1back = np.mean(slope_hbo_1back, axis=1).flatten()  # shape: (22, 1)
subjects_mean_hbr_1back = np.mean(mean_hbr_1back, axis=1).flatten()  # shape: (22, 1)
subjects_std_hbr_1back = np.mean(std_hbr_1back, axis=1).flatten()  # shape: (22, 1)
subjects_slope_hbr_1back = np.mean(slope_hbr_1back, axis=1).flatten()  # shape: (22, 1)
subjects_mean_eeg_1back = np.mean(features_eeg_1back, axis=1).flatten() # shape: (22,1)

# Compute the correlation matrix
correlation_matrix_1back = np.corrcoef(subjects_mean_hbo_1back, subjects_mean_eeg_1back, rowvar=False)  # rowvar = False means that each column represents a variable (subject)

# Print the correlation matrix
print("Correlation Matrix 1Back:")
print(correlation_matrix_1back)

# Create a df 

array_1Back = np.repeat(1, 22)

data_1Back = {
    "Mean Hbo": subjects_mean_hbo_1back,
    "Std Hbo": subjects_std_hbo_1back,
    "Slope Hbo": subjects_slope_hbo_1back,
    "Mean Hbr": subjects_mean_hbr_1back,
    "Std Hbr": subjects_std_hbr_1back,
    "Slope Hbr": subjects_slope_hbr_1back,
    "Theta Power Frontal Mid Line": subjects_mean_eeg_1back,
    "N_back" : array_1Back
}

# Create a DataFrame from the dictionary
df_1Back = pd.DataFrame(data_1Back)


# ----------------------------------------2 back--------------------------------------------------------------- 
path_hbo_2back = "H:\\Dokumenter\\data_processing\\Results_fNIRS\\features\\features_hbo_2Back.npy"
path_hbr_2back = "H:\\Dokumenter\\data_processing\\Results_fNIRS\\features\\features_hbr_2Back.npy"
path_eeg_2back = "H:\\Dokumenter\\data_processing\\Results_EEG\\features\\features_2Back.npy"

features_hbo_2back = np.load(path_hbo_2back)
features_hbr_2back = np.load(path_hbr_2back)
features_eeg_2back = np.load(path_eeg_2back)

# We take the mean of the HBO and only 22 patients to match dimensions with the EEG features
mean_hbo_2back = features_hbo_2back[0:22, :, 0]
mean_hbo_2back.reshape((22, 8, 1))

std_hbo_2back = features_hbo_2back[0:22, :, 1]
std_hbo_2back.reshape((22, 8, 1))

slope_hbo_2back = features_hbo_2back[0:22, :, 2]
slope_hbo_2back.reshape((22, 8, 1))

# We take the mean of the HBR and only 22 patients to match dimensions with the EEG features
mean_hbr_2back = features_hbr_2back[0:22, :, 0:1]
mean_hbr_2back.reshape((22, 8, 1))

std_hbr_2back = features_hbr_2back[0:22, :, 1:2]
std_hbr_2back.reshape((22, 8, 1))

slope_hbr_2back = features_hbr_2back[0:22, :, 2:3]
slope_hbr_2back.reshape((22, 8, 1))

# Compute the mean power across the channels for each subject
subjects_mean_hbo_2back = np.mean(mean_hbo_2back, axis=1).flatten()  # shape: (22, 1)
subjects_std_hbo_2back = np.mean(std_hbo_2back, axis=1).flatten()  # shape: (22, 1)
subjects_slope_hbo_2back = np.mean(slope_hbo_2back, axis=1).flatten()  # shape: (22, 1)
subjects_mean_hbr_2back = np.mean(mean_hbr_2back, axis=1).flatten()  # shape: (22, 1)
subjects_std_hbr_2back = np.mean(std_hbr_2back, axis=1).flatten()  # shape: (22, 1)
subjects_slope_hbr_2back = np.mean(slope_hbr_2back, axis=1).flatten()  # shape: (22, 1)
subjects_mean_eeg_2back = np.mean(features_eeg_2back, axis=1).flatten() # shape: (22,1)

# Compute the correlation matrix
correlation_matrix_2back = np.corrcoef(subjects_mean_hbo_2back, subjects_mean_eeg_2back, rowvar=False)  # rowvar = False means that each column represents a variable (subject)

# Print the correlation matrix
print("Correlation Matrix 2Back:")
print(correlation_matrix_2back)

# Create a df 

array_2Back = np.repeat(2, 22)

data_2Back = {
    "Mean Hbo": subjects_mean_hbo_2back,
    "Std Hbo": subjects_std_hbo_2back,
    "Slope Hbo": subjects_slope_hbo_2back,
    "Mean Hbr": subjects_mean_hbr_2back,
    "Std Hbr": subjects_std_hbr_2back,
    "Slope Hbr": subjects_slope_hbr_2back,
    "Theta Power Frontal Mid Line": subjects_mean_eeg_2back,
    "N_back" : array_2Back
}

# Create a DataFrame from the dictionary
df_2Back = pd.DataFrame(data_2Back)


# ----------------------------------------3 back--------------------------------------------------------------- 
path_hbo_3back = "H:\\Dokumenter\\data_processing\\Results_fNIRS\\features\\features_hbo_3Back.npy"
path_hbr_3back = "H:\\Dokumenter\\data_processing\\Results_fNIRS\\features\\features_hbr_3Back.npy"
path_eeg_3back = "H:\\Dokumenter\\data_processing\\Results_EEG\\features\\features_3Back.npy"

features_hbo_3back = np.load(path_hbo_3back)
features_hbr_3back = np.load(path_hbr_3back)
features_eeg_3back = np.load(path_eeg_3back)

# We take the mean of the HBO and only 22 patients to match dimensions with the EEG features
mean_hbo_3back = features_hbo_3back[0:22, :, 0]
mean_hbo_3back.reshape((22, 8, 1))

std_hbo_3back = features_hbo_3back[0:22, :, 1]
std_hbo_3back.reshape((22, 8, 1))

slope_hbo_3back = features_hbo_3back[0:22, :, 2]
slope_hbo_3back.reshape((22, 8, 1))

# We take the mean of the HBR and only 22 patients to match dimensions with the EEG features
mean_hbr_3back = features_hbr_3back[0:22, :, 0:1]
mean_hbr_3back.reshape((22, 8, 1))

std_hbr_3back = features_hbr_3back[0:22, :, 1:2]
std_hbr_3back.reshape((22, 8, 1))

slope_hbr_3back = features_hbr_3back[0:22, :, 2:3]
slope_hbr_3back.reshape((22, 8, 1))

# Compute the mean power across the channels for each subject
subjects_mean_hbo_3back = np.mean(mean_hbo_3back, axis=1).flatten()  # shape: (22, 1)
subjects_std_hbo_3back = np.mean(std_hbo_3back, axis=1).flatten()  # shape: (22, 1)
subjects_slope_hbo_3back = np.mean(slope_hbo_3back, axis=1).flatten()  # shape: (22, 1)
subjects_mean_hbr_3back = np.mean(mean_hbr_3back, axis=1).flatten()  # shape: (22, 1)
subjects_std_hbr_3back = np.mean(std_hbr_3back, axis=1).flatten()  # shape: (22, 1)
subjects_slope_hbr_3back = np.mean(slope_hbr_3back, axis=1).flatten()  # shape: (22, 1)
subjects_mean_eeg_3back = np.mean(features_eeg_3back, axis=1).flatten() # shape: (22,1)

# Compute the correlation matrix
correlation_matrix_3back = np.corrcoef(subjects_mean_hbo_3back, subjects_mean_eeg_0back, rowvar=False)  # rowvar = False means that each column represents a variable (subject)

# Print the correlation matrix
print("Correlation Matrix 3Back:")
print(correlation_matrix_3back)

# Create a df 

array_3Back = np.repeat(3, 22)

data_3Back = {
    "Mean Hbo": subjects_mean_hbo_3back,
    "Std Hbo": subjects_std_hbo_3back,
    "Slope Hbo": subjects_slope_hbo_3back,
    "Mean Hbr": subjects_mean_hbr_3back,
    "Std Hbr": subjects_std_hbr_3back,
    "Slope Hbr": subjects_slope_hbr_3back,
    "Theta Power Frontal Mid Line": subjects_mean_eeg_3back,
    "N_back" : array_3Back
}

# Create a DataFrame from the dictionary
df_3Back = pd.DataFrame(data_3Back)


df = pd.concat([df_0Back, df_1Back, df_2Back, df_3Back], axis=0)

df["N_back"] = df["N_back"].astype('category')
print(df.describe(include="category"))

columns = list(df.columns)
print(columns)

import seaborn as sns 
sns.scatterplot(x="Mean Hbo", y="Theta Power Frontal Mid Line",  data=df)

sns.lmplot(x="Mean Hbo", y="Theta Power Frontal Mid Line",  data=df)

sns.scatterplot(x="Mean Hbo", y="Theta Power Frontal Mid Line",  data=df)

sns.lmplot(x="Mean Hbo", y="Theta Power Frontal Mid Line", hue="N_back", data=df)

plt.show()

from scipy import stats 
hbo_theta = stats.pearsonr(df['Mean Hbo'], df['Theta Power Frontal Mid Line'])
hbr_theta = stats.pearsonr(df['Mean Hbr'], df['Theta Power Frontal Mid Line'])

print("This is hbo_theta", hbo_theta)
print("This is hbr_theta", hbr_theta)

cormat = df.corr()
print(round(cormat,2))

sns.heatmap(cormat)

plt.show()