import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ----------------------------------------0 back--------------------------------------------------------------- 
print("------------------------------------------------")
print("This is 0 back")

#path_hbo_0back = "/Users/adriana/Documents/DTU/thesis/data_processing/Results_fNIRS/features/0_back/features_hbo_0Back.npy"
#path_hbr_0back = "/Users/adriana/Documents/DTU/thesis/data_processing/Results_fNIRS/features/0_back/features_hbr_0Back.npy"
#path_eeg_0back = "/Users/adriana/Documents/DTU/thesis/data_processing/Results_EEG/healthy_controls/n_back/Relative PSD/0_back/0.npy"

path_hbo_0back = "H:\\Dokumenter\\data_processing\\Results_fNIRS\\features\\healthy_controls\\coupling_8\\0_back\\features_hbo_0Back.npy"
path_hbr_0back = "H:\\Dokumenter\\data_processing\\Results_fNIRS\\features\\healthy_controls\\coupling_8\\0_back\\features_hbr_0Back.npy"
path_eeg_0back = "H:\\Dokumenter\\data_processing\\Results EEG\\coupling_8\\healthy_controls\\n_back\\Relative PSD\\0_back\\0.npy"

features_hbo_0back = np.load(path_hbo_0back)
features_hbr_0back = np.load(path_hbr_0back)
features_eeg_0back = np.load(path_eeg_0back)

print("These are the sizes of the features", features_hbo_0back.shape, features_hbr_0back.shape, features_eeg_0back.shape)

theta_0back = features_eeg_0back[:, 1 ,: , :]
mean_hbo_0back = features_hbo_0back[:, :, :, 0]
mean_hbr_0back = features_hbr_0back[:,: , :, 0]

print("These are the sizes of the features (subjects, epochs, channels)", theta_0back.shape, mean_hbo_0back.shape, mean_hbr_0back.shape)

# Compute the mean power across the channels for each subject
mean_channels_hbo_0back = np.mean(mean_hbo_0back, axis=-1)  # shape: (subject, epochs)
mean_channels_hbr_0back = np.mean(mean_hbo_0back, axis=-1)  # shape: (subject, epochs)
mean_channels_theta_0back = np.mean(theta_0back, axis=-1)  # shape: (subject, epochs)

# Obtain a one dimensional array (1-D) with the flatten() function
#mean_channels_hbo_0back = np.mean(mean_hbo_0back, axis=-1).flatten()  # shape: (subject, epochs)
#mean_channels_hbr_0back = np.mean(mean_hbo_0back, axis=-1).flatten()  # shape: (subject, epochs)
#mean_channels_theta_0back = np.mean(theta_0back, axis=-1).flatten()  # shape: (subject, epochs)

print("These are the sizes of the features after averaging with the channels (subjects, epochs)", mean_channels_theta_0back.shape, 
      mean_channels_hbo_0back.shape, mean_channels_hbr_0back.shape)

# Obtain the average across all subjects
mean_sub_chan_hbo_0back = np.mean(mean_channels_hbo_0back, axis = 0) # shape (epochs, )
mean_sub_chan_hbr_0back = np.mean(mean_channels_hbr_0back, axis = 0) # shape (epochs, )
mean_sub_chan_theta_0back = np.mean(mean_channels_theta_0back, axis = 0) # shape (epochs, )

print("Size after averaging across all subjects (epochs, )", mean_sub_chan_theta_0back.shape, 
      mean_sub_chan_hbo_0back.shape, mean_sub_chan_hbr_0back.shape)


# ----------------------------------------1 back--------------------------------------------------------------- 
print("------------------------------------------------")
print("This is 1 back")

#path_hbo_1back = "/Users/adriana/Documents/DTU/thesis/data_processing/Results_fNIRS/features/1_back/features_hbo_1Back.npy"
#path_hbr_1back = "/Users/adriana/Documents/DTU/thesis/data_processing/Results_fNIRS/features/1_back/features_hbr_1Back.npy"
#path_eeg_1back = "/Users/adriana/Documents/DTU/thesis/data_processing/Results_EEG/healthy_controls/n_back/Relative PSD/1_back/1.npy"

path_hbo_1back = "H:\\Dokumenter\\data_processing\\Results_fNIRS\\features\\healthy_controls\\coupling_8\\1_back\\features_hbo_1Back.npy"
path_hbr_1back = "H:\\Dokumenter\\data_processing\\Results_fNIRS\\features\\healthy_controls\\coupling_8\\1_back\\features_hbr_1Back.npy"
path_eeg_1back = "H:\\Dokumenter\\data_processing\\Results EEG\\coupling_8\\healthy_controls\\n_back\\Relative PSD\\1_back\\1.npy"

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

print("These are the sizes of the features after averaging with the channels (subjects, epochs)", mean_channels_theta_1back.shape, 
      mean_channels_hbo_1back.shape, mean_channels_hbr_1back.shape)

# Obtain the average across all subjects
mean_sub_chan_hbo_1back = np.mean(mean_channels_hbo_1back, axis = 0) # shape (epochs, )
mean_sub_chan_hbr_1back = np.mean(mean_channels_hbr_1back, axis = 0) # shape (epochs, )
mean_sub_chan_theta_1back = np.mean(mean_channels_theta_1back, axis = 0) # shape (epochs, )

print("Size after averaging across all subjects (epochs, )", mean_sub_chan_theta_1back.shape, 
      mean_sub_chan_hbo_1back.shape, mean_sub_chan_hbr_1back.shape)

# ----------------------------------------2 back--------------------------------------------------------------- 
print("------------------------------------------------")
print("This is 2 back")

#path_hbo_2back = "/Users/adriana/Documents/DTU/thesis/data_processing/Results_fNIRS/features/2_back/features_hbo_2Back.npy"
#path_hbr_2back = "/Users/adriana/Documents/DTU/thesis/data_processing/Results_fNIRS/features/2_back/features_hbr_2Back.npy"
#path_eeg_2back = "/Users/adriana/Documents/DTU/thesis/data_processing/Results_EEG/healthy_controls/n_back/Relative PSD/2_back/2.npy"

path_hbo_2back = "H:\\Dokumenter\\data_processing\\Results_fNIRS\\features\\healthy_controls\\coupling_8\\2_back\\features_hbo_2Back.npy"
path_hbr_2back = "H:\\Dokumenter\\data_processing\\Results_fNIRS\\features\\healthy_controls\\coupling_8\\2_back\\features_hbr_2Back.npy"
path_eeg_2back = "H:\\Dokumenter\\data_processing\\Results EEG\\coupling_8\\healthy_controls\\n_back\\Relative PSD\\2_back\\2.npy"

features_hbo_2back = np.load(path_hbo_2back)
features_hbr_2back = np.load(path_hbr_2back)
features_eeg_2back = np.load(path_eeg_2back)

print("These are the sizes of the features", features_hbo_2back.shape, features_hbr_2back.shape, features_eeg_2back.shape)

theta_2back = features_eeg_2back[:, 1 ,: ,: ]
mean_hbo_2back = features_hbo_2back[: ,: ,: , 0]
mean_hbr_2back = features_hbr_2back[: ,: ,: , 0]

print("These are the sizes of the features (subjects, epochs, channels)", theta_2back.shape, mean_hbo_2back.shape, mean_hbr_2back.shape)

# Compute the mean power across the channels for each subject
mean_channels_hbo_2back = np.mean(mean_hbo_2back, axis=-1)  # shape: (subject, epochs)
mean_channels_hbr_2back = np.mean(mean_hbo_2back, axis=-1)  # shape: (subject, epochs)
mean_channels_theta_2back = np.mean(theta_2back, axis=-1)  # shape: (subject, epochs)

# Obtain a one dimensional array (1-D) with the flatten() function
#mean_channels_hbo_2back = np.mean(mean_hbo_2back, axis=-1).flatten()  # shape: (subject, epochs)
#mean_channels_hbr_2back = np.mean(mean_hbo_2back, axis=-1).flatten()  # shape: (subject, epochs)
#mean_channels_theta_2back = np.mean(theta_2back, axis=-1).flatten()  # shape: (subject, epochs)

print("These are the sizes of the features after averaging with the channels (subjects, epochs)", mean_channels_theta_2back.shape, 
      mean_channels_hbo_2back.shape, mean_channels_hbr_2back.shape)

# Obtain the average across all subjects
mean_sub_chan_hbo_2back = np.mean(mean_channels_hbo_2back, axis = 0) # shape (epochs, )
mean_sub_chan_hbr_2back = np.mean(mean_channels_hbr_2back, axis = 0) # shape (epochs, )
mean_sub_chan_theta_2back = np.mean(mean_channels_theta_2back, axis = 0) # shape (epochs, )

print("Size after averaging across all subjects (epochs, )", mean_sub_chan_theta_2back.shape, 
      mean_sub_chan_hbo_2back.shape, mean_sub_chan_hbr_2back.shape)


# ----------------------------------------3 back--------------------------------------------------------------- 
print("------------------------------------------------")
print("This is 3 back")

#path_hbo_3back = "/Users/adriana/Documents/DTU/thesis/data_processing/Results_fNIRS/features/3_back/features_hbo_3Back.npy"
#path_hbr_3back = "/Users/adriana/Documents/DTU/thesis/data_processing/Results_fNIRS/features/3_back/features_hbr_3Back.npy"
#path_eeg_3back = "/Users/adriana/Documents/DTU/thesis/data_processing/Results_EEG/healthy_controls/n_back/Relative PSD/3_back/3.npy"

path_hbo_3back = "H:\\Dokumenter\\data_processing\\Results_fNIRS\\features\\healthy_controls\\coupling_8\\3_back\\features_hbo_3Back.npy"
path_hbr_3back = "H:\\Dokumenter\\data_processing\\Results_fNIRS\\features\\healthy_controls\\coupling_8\\3_back\\features_hbr_3Back.npy"
path_eeg_3back = "H:\\Dokumenter\\data_processing\\Results EEG\\coupling_8\\healthy_controls\\n_back\\Relative PSD\\3_back\\3.npy"

features_hbo_3back = np.load(path_hbo_3back)
features_hbr_3back = np.load(path_hbr_3back)
features_eeg_3back = np.load(path_eeg_3back)

print("These are the sizes of the features", features_hbo_3back.shape, features_hbr_3back.shape, features_eeg_3back.shape)

theta_3back = features_eeg_3back[:, 1 ,: , :]
mean_hbo_3back = features_hbo_3back[: ,: ,: , 0]
mean_hbr_3back = features_hbr_3back[: ,: ,: , 0]

print("These are the sizes of the features (subjects, epochs, channels)", theta_3back.shape, mean_hbo_3back.shape, mean_hbr_3back.shape)

# Compute the mean power across the channels for each subject
mean_channels_hbo_3back = np.mean(mean_hbo_3back, axis=-1)  # shape: (subject, epochs)
mean_channels_hbr_3back = np.mean(mean_hbo_3back, axis=-1)  # shape: (subject, epochs)
mean_channels_theta_3back = np.mean(theta_3back, axis=-1)  # shape: (subject, epochs)

# Obtain a one dimensional array (1-D) with the flatten() function
#mean_channels_hbo_3back = np.mean(mean_hbo_3back, axis=-1).flatten()  # shape: (subject, epochs)
#mean_channels_hbr_3back = np.mean(mean_hbo_3back, axis=-1).flatten()  # shape: (subject, epochs)
#mean_channels_theta_3back = np.mean(theta_3back, axis=-1).flatten()  # shape: (subject, epochs)

print("These are the sizes of the features after averaging with the channels (subjects, epochs)", mean_channels_theta_3back.shape, 
      mean_channels_hbo_3back.shape, mean_channels_hbr_3back.shape)

# Obtain the average across all subjects
mean_sub_chan_hbo_3back = np.mean(mean_channels_hbo_3back, axis = 0) # shape (epochs, )
mean_sub_chan_hbr_3back = np.mean(mean_channels_hbr_3back, axis = 0) # shape (epochs, )
mean_sub_chan_theta_3back = np.mean(mean_channels_theta_3back, axis = 0) # shape (epochs, )

print("Size after averaging across all subjects (epochs, )", mean_sub_chan_theta_3back.shape, 
      mean_sub_chan_hbo_3back.shape, mean_sub_chan_hbr_3back.shape)
