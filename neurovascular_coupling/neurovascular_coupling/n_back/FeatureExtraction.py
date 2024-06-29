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
            mean_per_epoch_per_channel = np.expand_dims(np.mean(patient, axis=1), axis=1) # epochs x channels X mean
            
            # std of the concentration HbO and HbR
            std_per_epoch_per_channels = np.expand_dims(np.std(patient, axis=1), axis=1) # epoch x channels X std

            # not include the slope feature
            if id == 0:
                self.features = np.expand_dims(np.concatenate((mean_per_epoch_per_channel,std_per_epoch_per_channels), axis=1), axis=0) # 1 x epochs  x channels x 2 features (mean & std)
            else:
                tmp = np.expand_dims(np.concatenate((mean_per_epoch_per_channel,std_per_epoch_per_channels), axis=1), axis=0) # patients x epochs  x channels x 2 features (mean & std)
                self.features = np.concatenate((self.features, tmp), axis =0)


    def getFeatures(self):
        '''
        Get all the data in the form Patients X Events X Channels X Features. And the label/order of the events Patients X Events.
        '''

        return self.features

