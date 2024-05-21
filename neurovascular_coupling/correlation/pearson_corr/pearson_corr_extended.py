import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import scipy

from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

correlation = ["coupling_1", "coupling_2", "coupling_3", "coupling_4", "coupling_5", "coupling_6", "coupling_7", "coupling_8"]

# Create a figure and subplots
fig, axs = plt.subplots(4, 2, figsize=(12, 12))


print("------------------------------------------------")
print("This is 3 back")

corr_coeff_list = list()
corr_pvalue_list = list()

spearman_coeff_list = list()
spearman_pvalue_list = list()

for i in range(len(correlation)):
      path_hbo = "H:\\Dokumenter\\data_processing\\Results_fNIRS\\features\\{}\\{}\\{}\\features_hbo_{}Back.npy".format("healthy_controls", correlation[i], "3_back",3)
      path_hbr = "H:\\Dokumenter\\data_processing\\Results_fNIRS\\features\\{}\\{}\\{}\\features_hbr_{}Back.npy".format("healthy_controls", correlation[i], "3_back",3)
      path_eeg = "H:\\Dokumenter\\data_processing\\Results EEG\\{}\\{}\\n_back\\Relative PSD\\{}\\{}.npy".format(correlation[i], "healthy_controls", "3_back", 3)

      features_hbo = np.load(path_hbo)
      features_hbr = np.load(path_hbr)
      features_eeg = np.load(path_eeg)

      print("These are the sizes of the features", features_hbo.shape, features_hbr.shape, features_eeg.shape)

      theta = features_eeg[:, 1 , : , :] #(subjects, freq band, epochs, channels)
      mean_hbo = features_hbo[:, :, 0:4, 0] #(subjects, epochs, channels, feature)
      mean_hbr = features_hbr[:, :, 0:4, 0] #(subjects, epochs, channels, feature)

      print("These are the sizes of the features (subjects, epochs, channels)", theta.shape, mean_hbo.shape, mean_hbr.shape)

      # Compute the mean power across the channels for each subject
      mean_channels_hbo = np.mean(mean_hbo, axis=-1)  # shape: (subject, epochs)
      mean_channels_hbr = np.mean(mean_hbo, axis=-1)  # shape: (subject, epochs)
      mean_channels_theta = np.mean(theta, axis=-1)  # shape: (subject, epochs)

      # Obtain a one dimensional array (1-D) with the flatten() function
      #mean_channels_hbo = np.mean(mean_hbo, axis=-1).flatten()  # shape: (subject, epochs)
      #mean_channels_hbr = np.mean(mean_hbo, axis=-1).flatten()  # shape: (subject, epochs)
      #mean_channels_theta = np.mean(theta, axis=-1).flatten()  # shape: (subject, epochs)

      print("These are the sizes of the features after averaging with the channels (subjects, epochs)", mean_channels_theta.shape, 
            mean_channels_hbo.shape, mean_channels_hbr.shape)

      # Obtain the average across all subjects
      mean_sub_chan_hbo = np.mean(mean_channels_hbo, axis = 0) # shape (epochs, )
      mean_sub_chan_hbr = np.mean(mean_channels_hbr, axis = 0) # shape (epochs, )
      mean_sub_chan_theta = np.mean(mean_channels_theta, axis = 0) # shape (epochs, )

      a, b = np.polyfit(mean_sub_chan_theta, mean_sub_chan_hbo, 1)

      row = i // 2  # Determine subplot row
      col = i % 2   # Determine subplot column
      ax = axs[row, col]  # Get current subplot
      ax.set_title(correlation[i])  # Set title of subplot

      # Finding Pearson Correlation Coefficient 0 back
      corr_coef, p_value = pearsonr(mean_sub_chan_hbo, mean_sub_chan_theta)
      #print("Correlation Coefficient:", corr_coef)
      #print("p-value:", p_value)
      corr_coeff_list.append(corr_coef)

      spear_coef_0back, p_value_spear_0back = spearmanr(mean_sub_chan_hbo, mean_sub_chan_theta)
      spearman_coeff_list.append(spear_coef_0back)

      slope, intercept, r, p, stderr = scipy.stats.linregress(mean_sub_chan_hbo, mean_sub_chan_theta)

      line = f'Regression line: y={intercept: .2f} + {slope:.2f}x, r={r:.2f}'

      ax.plot(mean_sub_chan_hbo, mean_sub_chan_theta, linewidth=0, color= "r", marker = 's', label = 'Data points')
      ax.plot(mean_sub_chan_hbo, intercept + slope *  mean_sub_chan_hbo, label = line)
      ax.set_xlabel('HBO')
      ax.set_ylabel('theta power')
      ax.legend(facecolor='white')

# Adjust layout
plt.tight_layout()

# Show plot
plt.show()

print(corr_coeff_list)
print(spearman_coeff_list)
