import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

correlation = ["coupling_1", "coupling_2", "coupling_3", "coupling_4", "coupling_5", "coupling_6", "coupling_7", "coupling_8"]

# ----------------------------------------0 back--------------------------------------------------------------- 
print("------------------------------------------------")
print("This is 0 back")

corr_coeff_0back_list = list()
corr_pvalue_0back_list = list()

for i in range(len(correlation)):
      path_hbo_0back = "H:\\Dokumenter\\data_processing\\Results_fNIRS\\features\\healthy_controls\\{}\\0_back\\features_hbo_0Back.npy".format(correlation[i])
      path_hbr_0back = "H:\\Dokumenter\\data_processing\\Results_fNIRS\\features\\healthy_controls\\{}\\0_back\\features_hbr_0Back.npy".format(correlation[i])
      path_eeg_0back = "H:\\Dokumenter\\data_processing\\Results EEG\\{}\\healthy_controls\\n_back\\Relative PSD\\0_back\\0.npy".format(correlation[i])

      #path_hbo_0back = "/Users/adriana/Documents/DTU/thesis/data_processing/Results_fNIRS/features/0_back/features_hbo_0Back.npy"
      #path_hbr_0back = "/Users/adriana/Documents/DTU/thesis/data_processing/Results_fNIRS/features/0_back/features_hbr_0Back.npy"
      #path_eeg_0back = "/Users/adriana/Documents/DTU/thesis/data_processing/Results_EEG/healthy_controls/n_back/Relative PSD/0_back/0.npy"

      features_hbo_0back = np.load(path_hbo_0back)
      features_hbr_0back = np.load(path_hbr_0back)
      features_eeg_0back = np.load(path_eeg_0back)

      #print("These are the sizes of the features", features_hbo_0back.shape, features_hbr_0back.shape, features_eeg_0back.shape)

      theta_0back = features_eeg_0back[:, 1 ,: , :]
      mean_hbo_0back = features_hbo_0back[:, :, :, 0]
      mean_hbr_0back = features_hbr_0back[:,: , :, 0]

      #print("These are the sizes of the features (subjects, epochs, channels)", theta_0back.shape, mean_hbo_0back.shape, mean_hbr_0back.shape)

      # Compute the mean power across the channels for each subject
      mean_channels_hbo_0back = np.mean(mean_hbo_0back, axis=-1)  # shape: (subject, epochs)
      mean_channels_hbr_0back = np.mean(mean_hbo_0back, axis=-1)  # shape: (subject, epochs)
      mean_channels_theta_0back = np.mean(theta_0back, axis=-1)  # shape: (subject, epochs)

      # Obtain a one dimensional array (1-D) with the flatten() function
      #mean_channels_hbo_0back = np.mean(mean_hbo_0back, axis=-1).flatten()  # shape: (subject, epochs)
      #mean_channels_hbr_0back = np.mean(mean_hbo_0back, axis=-1).flatten()  # shape: (subject, epochs)
      #mean_channels_theta_0back = np.mean(theta_0back, axis=-1).flatten()  # shape: (subject, epochs)

      #print("These are the sizes of the features after averaging with the channels (subjects, epochs)", mean_channels_theta_0back.shape, 
      #      mean_channels_hbo_0back.shape, mean_channels_hbr_0back.shape)

      # Obtain the average across all subjects
      mean_sub_chan_hbo_0back = np.mean(mean_channels_hbo_0back, axis = 0) # shape (epochs, )
      mean_sub_chan_hbr_0back = np.mean(mean_channels_hbr_0back, axis = 0) # shape (epochs, )
      mean_sub_chan_theta_0back = np.mean(mean_channels_theta_0back, axis = 0) # shape (epochs, )

      #print("Size after averaging across all subjects (epochs, )", mean_sub_chan_theta_0back.shape, 
      #      mean_sub_chan_hbo_0back.shape, mean_sub_chan_hbr_0back.shape)
      
      # Finding Pearson Correlation Coefficient 0 back
      corr_coef_0back, p_value_0back = pearsonr(mean_sub_chan_hbo_0back, mean_sub_chan_theta_0back)
      #print("Correlation Coefficient 0 back:", corr_coef_0back)
      #print("p-value 0 back:", p_value_0back)

      corr_coeff_0back_list.append(corr_coef_0back)
      corr_pvalue_0back_list.append(p_value_0back)

print(corr_coeff_0back_list)
#print(corr_pvalue_0back_list)


# ----------------------------------------1 back--------------------------------------------------------------- 
print("------------------------------------------------")
print("This is 1 back")

corr_coeff_1back_list = list()
corr_pvalue_1back_list = list()

for i in range(len(correlation)):
      path_hbo_1back = "H:\\Dokumenter\\data_processing\\Results_fNIRS\\features\\healthy_controls\\{}\\1_back\\features_hbo_1Back.npy".format(correlation[i])
      path_hbr_1back = "H:\\Dokumenter\\data_processing\\Results_fNIRS\\features\\healthy_controls\\{}\\1_back\\features_hbr_1Back.npy".format(correlation[i])
      path_eeg_1back = "H:\\Dokumenter\\data_processing\\Results EEG\\{}\\healthy_controls\\n_back\\Relative PSD\\1_back\\1.npy".format(correlation[i])

      #path_hbo_1back = "/Users/adriana/Documents/DTU/thesis/data_processing/Results_fNIRS/features/1_back/features_hbo_1Back.npy"
      #path_hbr_1back = "/Users/adriana/Documents/DTU/thesis/data_processing/Results_fNIRS/features/1_back/features_hbr_1Back.npy"
      #path_eeg_1back = "/Users/adriana/Documents/DTU/thesis/data_processing/Results_EEG/healthy_controls/n_back/Relative PSD/1_back/1.npy"

      features_hbo_1back = np.load(path_hbo_1back)
      features_hbr_1back = np.load(path_hbr_1back)
      features_eeg_1back = np.load(path_eeg_1back)

      #print("These are the sizes of the features", features_hbo_1back.shape, features_hbr_1back.shape, features_eeg_1back.shape)

      theta_1back = features_eeg_1back[:, 1 ,: , :]
      mean_hbo_1back = features_hbo_1back[:, :, :, 0]
      mean_hbr_1back = features_hbr_1back[:,: , :, 0]

      #print("These are the sizes of the features (subjects, epochs, channels)", theta_1back.shape, mean_hbo_1back.shape, mean_hbr_1back.shape)

      # Compute the mean power across the channels for each subject
      mean_channels_hbo_1back = np.mean(mean_hbo_1back, axis=-1)  # shape: (subject, epochs)
      mean_channels_hbr_1back = np.mean(mean_hbo_1back, axis=-1)  # shape: (subject, epochs)
      mean_channels_theta_1back = np.mean(theta_1back, axis=-1)  # shape: (subject, epochs)

      # Obtain a one dimensional array (1-D) with the flatten() function
      #mean_channels_hbo_1back = np.mean(mean_hbo_1back, axis=-1).flatten()  # shape: (subject, epochs)
      #mean_channels_hbr_1back = np.mean(mean_hbo_1back, axis=-1).flatten()  # shape: (subject, epochs)
      #mean_channels_theta_1back = np.mean(theta_1back, axis=-1).flatten()  # shape: (subject, epochs)

      #print("These are the sizes of the features after averaging with the channels (subjects, epochs)", mean_channels_theta_1back.shape, 
      #      mean_channels_hbo_1back.shape, mean_channels_hbr_1back.shape)

      # Obtain the average across all subjects
      mean_sub_chan_hbo_1back = np.mean(mean_channels_hbo_1back, axis = 0) # shape (epochs, )
      mean_sub_chan_hbr_1back = np.mean(mean_channels_hbr_1back, axis = 0) # shape (epochs, )
      mean_sub_chan_theta_1back = np.mean(mean_channels_theta_1back, axis = 0) # shape (epochs, )

      #print("Size after averaging across all subjects (epochs, )", mean_sub_chan_theta_1back.shape, 
      #      mean_sub_chan_hbo_1back.shape, mean_sub_chan_hbr_1back.shape)
      
      # Finding Pearson Correlation Coefficient 1 back
      corr_coef_1back, p_value_1back = pearsonr(mean_sub_chan_hbo_1back, mean_sub_chan_theta_1back)
      #print("Correlation Coefficient 1 back:", corr_coef_1back)
      #print("p-value 1 back:", p_value_1back)

      corr_coeff_1back_list.append(corr_coef_1back)
      corr_pvalue_1back_list.append(p_value_1back)

print(corr_coeff_1back_list)
#print(corr_pvalue_1back_list)

# ----------------------------------------2 back--------------------------------------------------------------- 
print("------------------------------------------------")
print("This is 2 back")

corr_coeff_2back_list = list()
corr_pvalue_2back_list = list()

for i in range(len(correlation)):
      path_hbo_2back = "H:\\Dokumenter\\data_processing\\Results_fNIRS\\features\\healthy_controls\\{}\\2_back\\features_hbo_2Back.npy".format(correlation[i])
      path_hbr_2back = "H:\\Dokumenter\\data_processing\\Results_fNIRS\\features\\healthy_controls\\{}\\2_back\\features_hbr_2Back.npy".format(correlation[i])
      path_eeg_2back = "H:\\Dokumenter\\data_processing\\Results EEG\\{}\\healthy_controls\\n_back\\Relative PSD\\2_back\\2.npy".format(correlation[i])

      #path_hbo_2back = "/Users/adriana/Documents/DTU/thesis/data_processing/Results_fNIRS/features/2_back/features_hbo_2Back.npy"
      #path_hbr_2back = "/Users/adriana/Documents/DTU/thesis/data_processing/Results_fNIRS/features/2_back/features_hbr_2Back.npy"
      #path_eeg_2back = "/Users/adriana/Documents/DTU/thesis/data_processing/Results_EEG/healthy_controls/n_back/Relative PSD/2_back/2.npy"

      features_hbo_2back = np.load(path_hbo_2back)
      features_hbr_2back = np.load(path_hbr_2back)
      features_eeg_2back = np.load(path_eeg_2back)

      #print("These are the sizes of the features", features_hbo_2back.shape, features_hbr_2back.shape, features_eeg_2back.shape)

      theta_2back = features_eeg_2back[:, 1 ,: , :]
      mean_hbo_2back = features_hbo_2back[:, :, :, 0]
      mean_hbr_2back = features_hbr_2back[:,: , :, 0]

      #print("These are the sizes of the features (subjects, epochs, channels)", theta_2back.shape, mean_hbo_2back.shape, mean_hbr_2back.shape)

      # Compute the mean power across the channels for each subject
      mean_channels_hbo_2back = np.mean(mean_hbo_2back, axis=-1)  # shape: (subject, epochs)
      mean_channels_hbr_2back = np.mean(mean_hbo_2back, axis=-1)  # shape: (subject, epochs)
      mean_channels_theta_2back = np.mean(theta_2back, axis=-1)  # shape: (subject, epochs)

      # Obtain a one dimensional array (1-D) with the flatten() function
      #mean_channels_hbo_2back = np.mean(mean_hbo_2back, axis=-1).flatten()  # shape: (subject, epochs)
      #mean_channels_hbr_2back = np.mean(mean_hbo_2back, axis=-1).flatten()  # shape: (subject, epochs)
      #mean_channels_theta_2back = np.mean(theta_2back, axis=-1).flatten()  # shape: (subject, epochs)

      #print("These are the sizes of the features after averaging with the channels (subjects, epochs)", mean_channels_theta_2back.shape, 
      #      mean_channels_hbo_2back.shape, mean_channels_hbr_2back.shape)

      # Obtain the average across all subjects
      mean_sub_chan_hbo_2back = np.mean(mean_channels_hbo_2back, axis = 0) # shape (epochs, )
      mean_sub_chan_hbr_2back = np.mean(mean_channels_hbr_2back, axis = 0) # shape (epochs, )
      mean_sub_chan_theta_2back = np.mean(mean_channels_theta_2back, axis = 0) # shape (epochs, )

      #print("Size after averaging across all subjects (epochs, )", mean_sub_chan_theta_2back.shape, 
      #      mean_sub_chan_hbo_2back.shape, mean_sub_chan_hbr_2back.shape)
      
      # Finding Pearson Correlation Coefficient 2 back
      corr_coef_2back, p_value_2back = pearsonr(mean_sub_chan_hbo_2back, mean_sub_chan_theta_2back)
      #print("Correlation Coefficient 2 back:", corr_coef_2back)
      #print("p-value 2 back:", p_value_2back)

      corr_coeff_2back_list.append(corr_coef_2back)
      corr_pvalue_2back_list.append(p_value_2back)

print(corr_coeff_2back_list)
#print(corr_pvalue_2back_list)


# ----------------------------------------3 back--------------------------------------------------------------- 
print("------------------------------------------------")

print("This is 3 back")
corr_coeff_3back_list = list()
corr_pvalue_3back_list = list()

for i in range(len(correlation)):
      path_hbo_3back = "H:\\Dokumenter\\data_processing\\Results_fNIRS\\features\\healthy_controls\\{}\\3_back\\features_hbo_3Back.npy".format(correlation[i])
      path_hbr_3back = "H:\\Dokumenter\\data_processing\\Results_fNIRS\\features\\healthy_controls\\{}\\3_back\\features_hbr_3Back.npy".format(correlation[i])
      path_eeg_3back = "H:\\Dokumenter\\data_processing\\Results EEG\\{}\\healthy_controls\\n_back\\Relative PSD\\3_back\\3.npy".format(correlation[i])

      #path_hbo_3back = "/Users/adriana/Documents/DTU/thesis/data_processing/Results_fNIRS/features/3_back/features_hbo_3Back.npy"
      #path_hbr_3back = "/Users/adriana/Documents/DTU/thesis/data_processing/Results_fNIRS/features/3_back/features_hbr_3Back.npy"
      #path_eeg_3back = "/Users/adriana/Documents/DTU/thesis/data_processing/Results_EEG/healthy_controls/n_back/Relative PSD/3_back/3.npy"

      features_hbo_3back = np.load(path_hbo_3back)
      features_hbr_3back = np.load(path_hbr_3back)
      features_eeg_3back = np.load(path_eeg_3back)

      #print("These are the sizes of the features", features_hbo_3back.shape, features_hbr_3back.shape, features_eeg_3back.shape)

      theta_3back = features_eeg_3back[:, 1 ,: , :]
      mean_hbo_3back = features_hbo_3back[:, :, :, 0]
      mean_hbr_3back = features_hbr_3back[:,: , :, 0]

      #print("These are the sizes of the features (subjects, epochs, channels)", theta_3back.shape, mean_hbo_3back.shape, mean_hbr_3back.shape)

      # Compute the mean power across the channels for each subject
      mean_channels_hbo_3back = np.mean(mean_hbo_3back, axis=-1)  # shape: (subject, epochs)
      mean_channels_hbr_3back = np.mean(mean_hbo_3back, axis=-1)  # shape: (subject, epochs)
      mean_channels_theta_3back = np.mean(theta_3back, axis=-1)  # shape: (subject, epochs)

      # Obtain a one dimensional array (1-D) with the flatten() function
      #mean_channels_hbo_3back = np.mean(mean_hbo_3back, axis=-1).flatten()  # shape: (subject, epochs)
      #mean_channels_hbr_3back = np.mean(mean_hbo_3back, axis=-1).flatten()  # shape: (subject, epochs)
      #mean_channels_theta_3back = np.mean(theta_3back, axis=-1).flatten()  # shape: (subject, epochs)

      #print("These are the sizes of the features after averaging with the channels (subjects, epochs)", mean_channels_theta_3back.shape, 
      #      mean_channels_hbo_3back.shape, mean_channels_hbr_3back.shape)

      # Obtain the average across all subjects
      mean_sub_chan_hbo_3back = np.mean(mean_channels_hbo_3back, axis = 0) # shape (epochs, )
      mean_sub_chan_hbr_3back = np.mean(mean_channels_hbr_3back, axis = 0) # shape (epochs, )
      mean_sub_chan_theta_3back = np.mean(mean_channels_theta_3back, axis = 0) # shape (epochs, )

      #print("Size after averaging across all subjects (epochs, )", mean_sub_chan_theta_3back.shape, 
      #      mean_sub_chan_hbo_3back.shape, mean_sub_chan_hbr_3back.shape)
      
      # Finding Pearson Correlation Coefficient 3 back
      corr_coef_3back, p_value_3back = pearsonr(mean_sub_chan_hbo_3back, mean_sub_chan_theta_3back)
      #print("Correlation Coefficient 3 back:", corr_coef_3back)
      #print("p-value 3 back:", p_value_3back)

      corr_coeff_3back_list.append(corr_coef_3back)
      corr_pvalue_3back_list.append(p_value_3back)

print(corr_coeff_3back_list)
#print(corr_pvalue_3back_list)

