import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ----------------------------------------0 back--------------------------------------------------------------- 
path_hbo_0back = "/Users/adriana/Documents/DTU/thesis/data_processing/Results_fNIRS/features/0_back/features_hbo_0Back.npy"
path_hbr_0back = "/Users/adriana/Documents/DTU/thesis/data_processing/Results_fNIRS/features/0_back/features_hbr_0Back.npy"
path_eeg_0back = "/Users/adriana/Documents/DTU/thesis/data_processing/Results_EEG/healthy_controls/n_back/Relative PSD/0_back/0.npy"

features_hbo_0back = np.load(path_hbo_0back)
features_hbr_0back = np.load(path_hbr_0back)
features_eeg_0back = np.load(path_eeg_0back)

print("These are the sizes of the features", features_hbo_0back.shape, features_hbr_0back.shape, features_eeg_0back.shape)

theta_0back = features_eeg_0back[:, 1 ,: , :]
mean_hbo_0back = features_hbo_0back[:, :, :, 0]
mean_hbr_0back = features_hbr_0back[:, :, :, 0]

print("These are the sizes of the features (subjects, epochs, channels)", theta_0back.shape, mean_hbo_0back.shape, mean_hbr_0back.shape)

# Compute the mean power across the channels for each subject
mean_channels_hbo_0back = np.mean(mean_hbo_0back, axis=-1)  # shape: (subject, epochs)
mean_channels_hbr_0back = np.mean(mean_hbo_0back, axis=-1)  # shape: (subject, epochs)
mean_channels_theta_0back = np.mean(theta_0back, axis=-1)  # shape: (subject, epochs)

# Obtain a one dimensional array (1-D) with the flatten() function
#mean_channels_hbo_0back = np.mean(mean_hbo_0back, axis=-1).flatten()  # shape: (subject, epochs)
#mean_channels_hbr_0back = np.mean(mean_hbo_0back, axis=-1).flatten()  # shape: (subject, epochs)
#mean_channels_theta_0back = np.mean(theta_0back, axis=-1).flatten()  # shape: (subject, epochs)

print("These are the sizes of the features after averaging with the channels (subjects, epochs)", mean_channels_theta_0back.shape, mean_channels_hbo_0back.shape, mean_channels_hbr_0back.shape)

# ----------------------------------------1 back--------------------------------------------------------------- 
path_hbo_1back = "/Users/adriana/Documents/DTU/thesis/data_processing/Results_fNIRS/features/1_back/features_hbo_1Back.npy"
path_hbr_1back = "/Users/adriana/Documents/DTU/thesis/data_processing/Results_fNIRS/features/1_back/features_hbr_1Back.npy"
path_eeg_1back = "/Users/adriana/Documents/DTU/thesis/data_processing/Results_EEG/healthy_controls/n_back/Relative PSD/1_back/1.npy"

features_hbo_1back = np.load(path_hbo_1back)
features_hbr_1back = np.load(path_hbr_1back)
features_eeg_1back = np.load(path_eeg_1back)

print("These are the sizes of the features", features_hbo_1back.shape, features_hbr_1back.shape, features_eeg_1back.shape)

theta_1back = features_eeg_1back[:, 1 ,: , :]
mean_hbo_1back = features_hbo_1back[: ,: , :, 0]
mean_hbr_1back = features_hbr_1back[: ,: , :, 0]

print("These are the sizes of the features (subjects, epochs, channels)", theta_1back.shape, mean_hbo_1back.shape, mean_hbr_1back.shape)

# Compute the mean power across the channels for each subject
mean_channels_hbo_1back = np.mean(mean_hbo_1back, axis=-1)  # shape: (subject, epochs)
mean_channels_hbr_1back = np.mean(mean_hbo_1back, axis=-1)  # shape: (subject, epochs)
mean_channels_theta_1back = np.mean(theta_1back, axis=-1)  # shape: (subject, epochs)

# Obtain a one dimensional array (1-D) with the flatten() function
#mean_channels_hbo_1back = np.mean(mean_hbo_1back, axis=-1).flatten()  # shape: (subject, epochs)
#mean_channels_hbr_1back = np.mean(mean_hbo_1back, axis=-1).flatten()  # shape: (subject, epochs)
#mean_channels_theta_1back = np.mean(theta_1back, axis=-1).flatten()  # shape: (subject, epochs)

print("These are the sizes of the features after averaging with the channels (subjects, epochs)", mean_channels_theta_1back.shape, mean_channels_hbo_1back.shape, mean_channels_hbr_1back.shape)

# ----------------------------------------2 back--------------------------------------------------------------- 
path_hbo_2back = "/Users/adriana/Documents/DTU/thesis/data_processing/Results_fNIRS/features/2_back/features_hbo_2Back.npy"
path_hbr_2back = "/Users/adriana/Documents/DTU/thesis/data_processing/Results_fNIRS/features/2_back/features_hbr_2Back.npy"
path_eeg_2back = "/Users/adriana/Documents/DTU/thesis/data_processing/Results_EEG/healthy_controls/n_back/Relative PSD/2_back/2.npy"

features_hbo_2back = np.load(path_hbo_2back)
features_hbr_2back = np.load(path_hbr_2back)
features_eeg_2back = np.load(path_eeg_2back)

print("These are the sizes of the features", features_hbo_2back.shape, features_hbr_2back.shape, features_eeg_2back.shape)

theta_2back = features_eeg_2back[:, 1 , :, :]
mean_hbo_2back = features_hbo_2back[: ,: , :, 0]
mean_hbr_2back = features_hbr_2back[: ,: , :, 0]

print("These are the sizes of the features (subjects, epochs, channels)", theta_2back.shape, mean_hbo_2back.shape, mean_hbr_2back.shape)

# Compute the mean power across the channels for each subject
mean_channels_hbo_2back = np.mean(mean_hbo_2back, axis=-1)  # shape: (subject, epochs)
mean_channels_hbr_2back = np.mean(mean_hbo_2back, axis=-1)  # shape: (subject, epochs)
mean_channels_theta_2back = np.mean(theta_2back, axis=-1)  # shape: (subject, epochs)

# Obtain a one dimensional array (1-D) with the flatten() function
#mean_channels_hbo_2back = np.mean(mean_hbo_2back, axis=-1).flatten()  # shape: (subject, epochs)
#mean_channels_hbr_2back = np.mean(mean_hbo_2back, axis=-1).flatten()  # shape: (subject, epochs)
#mean_channels_theta_2back = np.mean(theta_2back, axis=-1).flatten()  # shape: (subject, epochs)

print("These are the sizes of the features after averaging with the channels (subjects, epochs)", mean_channels_theta_2back.shape, mean_channels_hbo_2back.shape, mean_channels_hbr_2back.shape)


# ----------------------------------------3 back--------------------------------------------------------------- 
path_hbo_3back = "/Users/adriana/Documents/DTU/thesis/data_processing/Results_fNIRS/features/3_back/features_hbo_3Back.npy"
path_hbr_3back = "/Users/adriana/Documents/DTU/thesis/data_processing/Results_fNIRS/features/3_back/features_hbr_3Back.npy"
path_eeg_3back = "/Users/adriana/Documents/DTU/thesis/data_processing/Results_EEG/healthy_controls/n_back/Relative PSD/3_back/3.npy"

features_hbo_3back = np.load(path_hbo_3back)
features_hbr_3back = np.load(path_hbr_3back)
features_eeg_3back = np.load(path_eeg_3back)

print("These are the sizes of the features", features_hbo_3back.shape, features_hbr_3back.shape, features_eeg_3back.shape)

theta_3back = features_eeg_3back[:, 1 ,: , :]
mean_hbo_3back = features_hbo_3back[: ,: , :, 0]
mean_hbr_3back = features_hbr_3back[: ,:, :, 0]

print("These are the sizes of the features (subjects, epochs, channels)", theta_3back.shape, mean_hbo_3back.shape, mean_hbr_3back.shape)

# Compute the mean power across the channels for each subject
mean_channels_hbo_3back = np.mean(mean_hbo_3back, axis=-1)  # shape: (subject, epochs)
mean_channels_hbr_3back = np.mean(mean_hbo_3back, axis=-1)  # shape: (subject, epochs)
mean_channels_theta_3back = np.mean(theta_3back, axis=-1)  # shape: (subject, epochs)

# Obtain a one dimensional array (1-D) with the flatten() function
#mean_channels_hbo_3back = np.mean(mean_hbo_3back, axis=-1).flatten()  # shape: (subject, epochs)
#mean_channels_hbr_3back = np.mean(mean_hbo_3back, axis=-1).flatten()  # shape: (subject, epochs)
#mean_channels_theta_3back = np.mean(theta_3back, axis=-1).flatten()  # shape: (subject, epochs)

print("These are the sizes of the features after averaging with the channels (subjects, epochs)", mean_channels_theta_3back.shape, mean_channels_hbo_3back.shape, mean_channels_hbr_3back.shape)

# 0 BACK - PCA dimensionality reduction

# HBO
mean_channels_hbo_0back_transpose = np.transpose(mean_channels_hbo_0back)

plt.scatter(mean_channels_hbo_0back_transpose[:,0], mean_channels_hbo_0back_transpose[:,1])
plt.show()

scaler_hbo_0back = StandardScaler()
X_std_hbo_0back = scaler_hbo_0back.fit_transform(mean_channels_hbo_0back_transpose)

# Step 2-5: PCA
pca_hbo_0back = PCA()
X_pca_hbo_0back = pca_hbo_0back.fit_transform(X_std_hbo_0back).flatten()
X_pca_hbo_0back_plot = pca_hbo_0back.fit_transform(X_std_hbo_0back)

print("This is pca_hbo_0back", X_pca_hbo_0back)

#print("These are the PCA HBO components",pca_hbo_0back.components_)
#print("This is the PCA explained variance", pca_hbo_0back.explained_variance_)

X_new = pca_hbo_0back.inverse_transform(X_pca_hbo_0back_plot)
plt.scatter(X_std_hbo_0back[:, 0], X_std_hbo_0back[:, 1], alpha=0.2)
plt.scatter(X_new[:, 0], X_new[:, 1], alpha=0.8)
plt.axis('equal')
plt.show()

# EEG
mean_channels_theta_0back_transpose = np.transpose(mean_channels_theta_0back)
scaler_eeg_0back = StandardScaler()
X_std_eeg_0back = scaler_eeg_0back.fit_transform(mean_channels_theta_0back_transpose)

# Step 2-5: PCA
pca_eeg_0back = PCA()
X_pca_eeg_0back = pca_eeg_0back.fit_transform(X_std_eeg_0back).flatten()

#print("These are the PCA EEG components",pca_eeg_0back.components_)
#print("This is the PCA explained variance", pca_eeg_0back.explained_variance_)

print("This is X_pca_hbo_0back", X_pca_eeg_0back)

# CORRELATION
# for the correlation is necessary that we have a 1d array

# Finding Pearson Correlation Coefficient 0 back
corr_coef_0back, p_value_0back = pearsonr(X_pca_hbo_0back, X_pca_eeg_0back)
print("Correlation Coefficient 0 back:", corr_coef_0back)
print("p-value 0 back:", p_value_0back)

# 1 BACK - PCA dimensionality reduction

#plt.scatter(mean_channels_hbo_0back[:,0], mean_channels_theta_0back[:,1])
#plt.show()

# Step 1: Standardize the Data
mean_channels_hbo_1back_transpose = np.transpose(mean_channels_hbo_0back)
scaler_hbo_1back = StandardScaler()
X_std_hbo_1back = scaler_hbo_1back.fit_transform(mean_channels_hbo_1back_transpose)

# Step 2-5: PCA
pca_hbo_1back = PCA(n_components=2)
X_pca_hbo_1back = pca_hbo_1back.fit_transform(X_std_hbo_1back).flatten()

print("This is Xpca_hbo_1back", X_pca_hbo_1back)

#print("These are the PCA HBO components",pca_hbo_0back.components_)
#print("This is the PCA explained variance", pca_hbo_0back.explained_variance_)

#X_new = pca_hbo_0back.inverse_transform(X_pca_hbo_0back)
#plt.scatter(X_std_hbo_0back[:, 0], X_std_hbo_0back[:, 1], alpha=0.2)
#plt.scatter(X_new[:, 0], X_new[:, 1], alpha=0.8)
#plt.axis('equal')
#plt.show()

# Step 1: Standardize the Data
mean_channels_theta_1back_transpose = np.transpose(mean_channels_theta_0back)
scaler_eeg_1back = StandardScaler()
X_std_eeg_1back = scaler_eeg_1back.fit_transform(mean_channels_theta_0back_transpose)

# Step 2-5: PCA
pca_eeg_1back = PCA(n_components = 2)
X_pca_eeg_1back = pca_eeg_1back.fit_transform(X_std_eeg_1back).flatten()

#print("These are the PCA EEG components",pca_eeg_0back.components_)
#print("This is the PCA explained variance", pca_eeg_0back.explained_variance_)

print("This is X_pca_hbo_1back", X_pca_eeg_1back)

# CORRELATION
# for the correlation is necessary that we have a 1d array

# Finding Pearson Correlation Coefficient 0 back
corr_coef_1back, p_value_1back = pearsonr(X_pca_hbo_1back, X_pca_eeg_1back)
print("Correlation Coefficient 1 back:", corr_coef_1back)
print("p-value 1 back:", p_value_1back)

