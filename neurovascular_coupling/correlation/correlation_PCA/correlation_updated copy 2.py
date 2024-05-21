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

df_hbo = pd.DataFrame(mean_channels_hbo_0back)
print(df_hbo)

for c in df_hbo.columns:
    mean = df_hbo[c].mean() # mean of each column
    std = df_hbo[c].std() # standard deviation of each column
    df_hbo[c] = (df_hbo[c]-mean)/std # standardizing the data
print(df_hbo)

'''Covariance matrix of data'''
cov_matrix = np.cov(df_hbo, rowvar=False)

print(cov_matrix)

'''Calculate eigen values and eigen vectors'''

eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
eigenvalues_sorted = sorted(eigenvalues,reverse=True) # sort eigenvalues in decsnding order
eigenvectors_sorted = np.sort(eigenvectors)[::-1] # sort eigenvectors in descending order

print(f' Sorted eigen values:\n\n {eigenvalues_sorted} \n\n Sorted eigen vectors:\n\n{eigenvectors_sorted}')

'''Variance expalined by eigen values'''

var_explained = [eigen_value/sum(eigenvalues) for eigen_value in eigenvalues_sorted]
cumularive_variance = np.cumsum(var_explained) #calculate cumulative variance

'''Plot cumulative variance'''
plt.subplots(figsize=(4,4))
plt.plot(cumularive_variance,marker='s',markerfacecolor='w')
plt.grid()
for i in range(len(var_explained)):
    plt.axvline(x=i,linestyle=':',color='orange')
plt.title('Cumulative variance') 
plt.xlabel('Number of principal components')
plt.ylabel('Cumulative variance explained')
plt.show()

first3_eigenvectors = eigenvectors[:,:3]
principal_omponents = np.dot(df_hbo,first3_eigenvectors)
print(f'first 3 principal componenets corresponding to data:\n\n{principal_omponents}')

pca = PCA(n_components=2) # make an instance of PCA with 3 principle components
principal_components = pca.fit_transform(df_hbo) #fit the data to PCA and get components
print(f'first 2 principal components using scikit-learn:\n\n{principal_components}')

correlation_matrix = np.corrcoef(principal_components, rowvar=False)
print(f'correlation_matrix:\n\n {correlation_matrix}\n\n correlation matrix rounded to 15 decimals:{np.round(correlation_matrix,15)}')


df_eeg = pd.DataFrame(mean_channels_theta_0back)
print(df_eeg)

for c in df_eeg.columns:
    mean = df_eeg[c].mean() # mean of each column
    std = df_eeg[c].std() # standard deviation of each column
    df_eeg[c] = (df_eeg[c]-mean)/std # standardizing the data
print(df_eeg)

'''Covariance matrix of data'''
cov_matrix = np.cov(df_eeg, rowvar=False)

print(cov_matrix)

'''Calculate eigen values and eigen vectors'''

eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
eigenvalues_sorted = sorted(eigenvalues,reverse=True) # sort eigenvalues in decsnding order
eigenvectors_sorted = np.sort(eigenvectors)[::-1] # sort eigenvectors in descending order

print(f' Sorted eigen values:\n\n {eigenvalues_sorted} \n\n Sorted eigen vectors:\n\n{eigenvectors_sorted}')

'''Variance expalined by eigen values'''

var_explained = [eigen_value/sum(eigenvalues) for eigen_value in eigenvalues_sorted]
cumularive_variance = np.cumsum(var_explained) #calculate cumulative variance

'''Plot cumulative variance'''
plt.subplots(figsize=(4,4))
plt.plot(cumularive_variance,marker='s',markerfacecolor='w')
plt.grid()
for i in range(len(var_explained)):
    plt.axvline(x=i,linestyle=':',color='orange')
plt.title('Cumulative variance') 
plt.xlabel('Number of principal components')
plt.ylabel('Cumulative variance explained')
plt.show()

first3_eigenvectors = eigenvectors[:,:3]
principal_omponents = np.dot(df_eeg,first3_eigenvectors)
print(f'first 3 principal componenets corresponding to data:\n\n{principal_omponents}')

pca = PCA(n_components=2) # make an instance of PCA with 3 principle components
principal_components_eeg = pca.fit_transform(df_eeg) #fit the data to PCA and get components
print(f'first 2 principal components using scikit-learn:\n\n{principal_components_eeg}')

correlation_matrix = np.corrcoef(principal_components_eeg, rowvar=False)
print(f'correlation_matrix:\n\n {correlation_matrix}\n\n correlation matrix rounded to 15 decimals:{np.round(correlation_matrix,15)}')


correlation_matrix_good = np.corrcoef(principal_components, principal_components_eeg, rowvar=False)
print(f'correlation_matrix:\n\n {correlation_matrix_good}\n\n correlation matrix rounded to 15 decimals:{np.round(correlation_matrix_good,15)}')