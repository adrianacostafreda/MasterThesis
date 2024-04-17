import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import sklearn as sk
from sklearn import decomposition, preprocessing
from matplotlib.colors import ListedColormap

from sklearn.preprocessing import StandardScaler


#path_hc_baseline_eeg="/Users/adriana/Documents/GitHub/MasterThesis/"
path_hc_baseline_eeg="H:\\Dokumenter\\Visual_Studio\\data_processing_eeg\\.venv\\"

# ---- Load ----
df = pd.read_csv(os.path.join(path_hc_baseline_eeg, "eeg_features.csv"))

# First we will do it without baseline 

#baseline_df = df[df['baseline'] == "yes"]
trigger_df = df[df['baseline'] == 0]
del df


for index, row in trigger_df.iloc[1:].iterrows():

    id = row['patient_id']
    nback = row['n-back']
        
    # Without baseline
    delta_power = trigger_df.at[index,'delta']
    theta_power = trigger_df.at[index,'theta'] 
    alpha_power = trigger_df.at[index,'alpha']
    se = trigger_df.at[index,'se']
    pe = trigger_df.at[index,'pe']
    zc = trigger_df.at[index,'zc']

    print("This is delta power", delta_power)

temp = []

for feature in trigger_df.columns:
    print(feature)
    feature_vector = trigger_df[feature].values # save in a feature vector all the values of one column

    ## Visualize data distribution before 
    #plt.figure()
    #plt.hist(feature_vector, bins = 100)
    #plt.title("Data distribution for feature: {}".format(feature))
    #plt.show()

    temp.append(feature_vector)

X = np.column_stack(temp)

# Normalize each feature in range [0,1]

X = preprocessing.normalize(X, axis = 1)

# Dimensionality reduction with PCA
plt.cla()
pca_components = 2

pca = decomposition.PCA(n_components=pca_components)
pca.fit(X)
X = pca.transform(X)

for i in range(pca_components):
    print("Coordinates of principal component {} in feature space: {}".format(i,pca.components_[i]))

# Visualize PCA transformed data 
x = X[:,0]
y = X[:, 1]
labels = trigger_df['n-back'].values
plt.scatter(x, y, marker="o", c=labels, cmap=ListedColormap(['red','blue','green','orange']))
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(["0-back","1-back","2-back","3-back"])
plt.grid()
plt.show()

# save 
# np.save(os.path.join(path_hc_baseline_eeg,"X"), X)
# np.save(os.path.join(path_hc_baseline_eeg,"labels"), labels)





