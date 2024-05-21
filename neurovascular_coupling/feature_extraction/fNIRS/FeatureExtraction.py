import math
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

class FeatureExtraction():
    """
    """
    def __init__(self, data_path: str) -> None:
        """
        """
        if (data_path == "" or data_path is None):
            print("Specify the path of the raw data")
            return None
        
        self.data = np.load(data_path) # patients x epochs x channels x samples

        self.features = None # patients x channels x features

        for id, patient in enumerate(self.data):

            # mean of the concentration
            mean_per_epoch_per_channel = np.expand_dims(np.mean(patient, axis=2), axis=2) # epochs x channels X mean
            
            # std of the concentration HbO and HbR
            std_per_epoch_per_channels = np.expand_dims(np.std(patient, axis=2), axis=2) # epoch x channels X std

            # not include the slope feature
            if id == 0:
                self.features = np.expand_dims(np.concatenate((mean_per_epoch_per_channel,std_per_epoch_per_channels), axis=2), axis=0) # 1 x epochs  x channels x 2 features (mean & std)
            else:
                tmp = np.expand_dims(np.concatenate((mean_per_epoch_per_channel,std_per_epoch_per_channels), axis=2), axis=0) # patients x epochs  x channels x 2 features (mean & std)
                self.features = np.concatenate((self.features, tmp), axis =0)


    def getFeatures(self):
        '''
        Get all the data in the form Patients X Events X Channels X Features. And the label/order of the events Patients X Events.
        '''

        return self.features


# 0back
# Paths to your data 
#data_path_hbo_0back = "/Users/adriana/Documents/DTU/thesis/data_processing/Results_fNIRS/0_back/hemo_0back.npy"
#data_path_hbr_0back = "/Users/adriana/Documents/DTU/thesis/data_processing/Results_fNIRS/0_back/dehemo_0back.npy"
data_path_hbo_0back = "H:\\Dokumenter\\data_processing\\Results_fNIRS\\patients\\coupling_8\\0_back\\hemo_0back.npy"
data_path_hbr_0back = "H:\\Dokumenter\\data_processing\\Results_fNIRS\\patients\\coupling_8\\0_back\\dehemo_0back.npy"

# Create an instance of FeatureExtraction HB0
feature_extractor_hbo_0back = FeatureExtraction(data_path_hbo_0back)
feature_extractor_hbr_0back = FeatureExtraction(data_path_hbr_0back)

# Get features and labels
features_hbo_0back = feature_extractor_hbo_0back.getFeatures()
features_hbr_0back = feature_extractor_hbr_0back.getFeatures()

print("This is the shape of features_hbo_0back", features_hbo_0back.shape)
print("This is the shape of features_hbr_0back", features_hbr_0back.shape)

#clean_path_features_0back = "/Users/adriana/Documents/DTU/thesis/data_processing/Results_fNIRS/features/0_back/"
clean_path_features_0back = "H:\\Dokumenter\\data_processing\\Results_fNIRS\\features\\patients\\coupling_8\\0_back\\"

np.save(clean_path_features_0back + "features_hbo_0Back", features_hbo_0back)
np.save(clean_path_features_0back + "features_hbr_0Back", features_hbr_0back)


# 1back
# Paths to your data 
#data_path_hbo_1back = "/Users/adriana/Documents/DTU/thesis/data_processing/Results_fNIRS/1_back/hemo_1back.npy"
#data_path_hbr_1back = "/Users/adriana/Documents/DTU/thesis/data_processing/Results_fNIRS/1_back/dehemo_1back.npy"
data_path_hbo_1back = "H:\\Dokumenter\\data_processing\\Results_fNIRS\\patients\\coupling_8\\1_back\\hemo_1back.npy"
data_path_hbr_1back = "H:\\Dokumenter\\data_processing\\Results_fNIRS\\patients\\coupling_8\\1_back\\dehemo_1back.npy"

# Create an instance of FeatureExtraction HB0
feature_extractor_hbo_1back = FeatureExtraction(data_path_hbo_1back)
feature_extractor_hbr_1back = FeatureExtraction(data_path_hbr_1back)

# Get features and labels
features_hbo_1back = feature_extractor_hbo_1back.getFeatures()
features_hbr_1back = feature_extractor_hbr_1back.getFeatures()

print("This is the shape of features_hbo_1back", features_hbo_1back.shape)
print("This is the shape of features_hbr_1back", features_hbr_1back.shape)

#clean_path_features_1back = "/Users/adriana/Documents/DTU/thesis/data_processing/Results_fNIRS/features/1_back/"
clean_path_features_1back = "H:\\Dokumenter\\data_processing\\Results_fNIRS\\features\\patients\\coupling_8\\1_back\\"

np.save(clean_path_features_1back + "features_hbo_1Back", features_hbo_1back)
np.save(clean_path_features_1back + "features_hbr_1Back", features_hbr_1back)


# 2back
# Paths to your data 
#data_path_hbo_2back = "/Users/adriana/Documents/DTU/thesis/data_processing/Results_fNIRS/2_back/hemo_2back.npy"
#data_path_hbr_2back = "/Users/adriana/Documents/DTU/thesis/data_processing/Results_fNIRS/2_back/dehemo_2back.npy"
data_path_hbo_2back = "H:\\Dokumenter\\data_processing\\Results_fNIRS\\patients\\coupling_8\\2_back\\hemo_2back.npy"
data_path_hbr_2back = "H:\\Dokumenter\\data_processing\\Results_fNIRS\\patients\\coupling_8\\2_back\\dehemo_2back.npy"

# Create an instance of FeatureExtraction HB0
feature_extractor_hbo_2back = FeatureExtraction(data_path_hbo_2back)
feature_extractor_hbr_2back = FeatureExtraction(data_path_hbr_2back)

# Get features and labels
features_hbo_2back = feature_extractor_hbo_2back.getFeatures()
features_hbr_2back = feature_extractor_hbr_2back.getFeatures()

print("This is the shape of features_hbo_2back", features_hbo_2back.shape)
print("This is the shape of features_hbr_2back", features_hbr_2back.shape)

#clean_path_features_2back = "/Users/adriana/Documents/DTU/thesis/data_processing/Results_fNIRS/features/2_back/"
clean_path_features_2back = "H:\\Dokumenter\\data_processing\\Results_fNIRS\\features\\patients\\coupling_8\\2_back\\"

np.save(clean_path_features_2back + "features_hbo_2Back", features_hbo_2back)
np.save(clean_path_features_2back + "features_hbr_2Back", features_hbr_2back)


# 3back
# Paths to your data 
#data_path_hbo_3back = "/Users/adriana/Documents/DTU/thesis/data_processing/Results_fNIRS/3_back/hemo_3back.npy"
#data_path_hbr_3back = "/Users/adriana/Documents/DTU/thesis/data_processing/Results_fNIRS/3_back/dehemo_3back.npy"
data_path_hbo_3back = "H:\\Dokumenter\\data_processing\\Results_fNIRS\\patients\\coupling_8\\3_back\\hemo_3back.npy"
data_path_hbr_3back = "H:\\Dokumenter\\data_processing\\Results_fNIRS\\patients\\coupling_8\\3_back\\dehemo_3back.npy"

# Create an instance of FeatureExtraction HB0
feature_extractor_hbo_3back = FeatureExtraction(data_path_hbo_3back)
feature_extractor_hbr_3back = FeatureExtraction(data_path_hbr_3back)

# Get features and labels
features_hbo_3back = feature_extractor_hbo_3back.getFeatures()
features_hbr_3back = feature_extractor_hbr_3back.getFeatures()

print("This is the shape of features_hbo_3back", features_hbo_3back.shape)
print("This is the shape of features_hbr_3back", features_hbr_3back.shape)

#clean_path_features_3back = "/Users/adriana/Documents/DTU/thesis/data_processing/Results_fNIRS/features/3_back/"
clean_path_features_3back = "H:\\Dokumenter\\data_processing\\Results_fNIRS\\features\\patients\\coupling_8\\3_back\\"

np.save(clean_path_features_3back + "features_hbo_3Back", features_hbo_3back)
np.save(clean_path_features_3back + "features_hbr_3Back", features_hbr_3back)
