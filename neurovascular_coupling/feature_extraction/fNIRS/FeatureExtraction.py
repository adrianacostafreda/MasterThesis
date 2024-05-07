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
    def __init__(self, data_path: str, label_path: str) -> None:
        """
        """
        if (data_path == "" or data_path is None) or (label_path == "" or label_path is None):
            print("Specify the path of the raw data and the labels.")
            return None
        
        self.data = np.load(data_path) # patients X events X channels X samples
        self.labels = np.load(label_path)

        self.features = None # patients X events X channels X features

        for id, patient in enumerate(self.data):

            # mean of the concentration
            mean_per_event_per_channel = np.expand_dims(np.mean(patient, axis=2), axis=2) # events x channels X mean
            
            # std of the concentration HbO and HbR
            std_per_event_per_channels = np.expand_dims(np.std(patient, axis=2), axis=2) # events x channels X std

            """#Calculate slope for each event for each channel: events X channels X 1 slope
            slope_per_event_per_channel = list()
            for iid, event in enumerate(patient):
                slope_per_event = list()
                for iiid, channel_data in enumerate(event):
                    slope_per_event.append(np.polyfit(np.arange(channel_data.shape[0]),channel_data,1)[0])
                slope_per_event = np.expand_dims( np.array(slope_per_event), axis=1) # channels x 1 

                if iid == 0:
                    slope_per_event_per_channel = np.expand_dims(slope_per_event,axis=0) # events x channels X 1
                else:
                    slope_per_event_per_channel = np.concatenate((slope_per_event_per_channel,np.expand_dims(slope_per_event,axis=0)), axis=0) # events x channels X 1
            
            if id == 0:
                self.features = np.expand_dims(np.concatenate((mean_per_event_per_channel,std_per_event_per_channels,slope_per_event_per_channel), axis=2), axis=0) # 1 x events  x channels x 3 features
            else:
                tmp = np.expand_dims(np.concatenate((mean_per_event_per_channel,std_per_event_per_channels,slope_per_event_per_channel), axis=2), axis=0) # patients x events  x channels x 3 features
                self.features = np.concatenate((self.features, tmp), axis =0)
            """
            # not include the slope feature
            if id == 0:
                self.features = np.expand_dims(np.concatenate((mean_per_event_per_channel,std_per_event_per_channels), axis=2), axis=0) # 1 x events  x channels x 2 features (mean & std)
            else:
                tmp = np.expand_dims(np.concatenate((mean_per_event_per_channel,std_per_event_per_channels), axis=2), axis=0) # patients x events  x channels x 2 features (mean & std)
                self.features = np.concatenate((self.features, tmp), axis =0)


    def getFeatures(self):
        '''
        Get all the data in the form Patients X Events X Channels X Features. And the label/order of the events Patients X Events.
        '''

        return self.features, self.labels


# Paths to your data and labels
data_path_hbo = "H:\\Dokumenter\\data_processing\\Results_fNIRS\\patients\\hemo_per_event.npy"
label_path_hbo = "H:\\Dokumenter\\data_processing\\Results_fNIRS\\patients\\event_order_hbo.npy"

# Create an instance of FeatureExtraction HB0
feature_extractor_hbo = FeatureExtraction(data_path_hbo, label_path_hbo)

# Get features and labels
features_hbo, labels_hbo = feature_extractor_hbo.getFeatures()

features_hbo_0back = features_hbo[:,0,:,:]
features_hbo_1back = features_hbo[:,1,:,:]
features_hbo_2back = features_hbo[:,2,:,:]
features_hbo_3back = features_hbo[:,3,:,:]

data_path_hbr = "H:\\Dokumenter\\data_processing\\Results_fNIRS\\patients\\dehemo_per_event.npy"
label_path_hbr = "H:\\Dokumenter\\data_processing\\Results_fNIRS\\patients\\event_order_hbr.npy"

# Create an instance of FeatureExtraction HBR
feature_extractor_hbr = FeatureExtraction(data_path_hbr, label_path_hbr)

# Get features and labels
features_hbr, labels_hbr = feature_extractor_hbr.getFeatures()

features_hbr_0back = features_hbr[:,0,:,:]
features_hbr_1back = features_hbr[:,1,:,:]
features_hbr_2back = features_hbr[:,2,:,:]
features_hbr_3back = features_hbr[:,3,:,:]

print(features_hbr_0back.shape)

# save the features in numpy arrays

#clean_path = "/Users/adriana/Documents/DTU/thesis/data_processing/Results_fNIRS/"
clean_path = "H:\\Dokumenter\\data_processing\\Results_fNIRS\\features\\patients\\"

# HBO

np.save(clean_path + "features_hbo_0Back", features_hbo_0back)
np.save(clean_path + "features_hbo_1Back", features_hbo_1back)
np.save(clean_path + "features_hbo_2Back", features_hbo_2back)
np.save(clean_path + "features_hbo_3Back", features_hbo_3back)

#HBR
np.save(clean_path + "features_hbr_0Back", features_hbr_0back)
np.save(clean_path + "features_hbr_1Back", features_hbr_1back)
np.save(clean_path + "features_hbr_2Back", features_hbr_2back)
np.save(clean_path + "features_hbr_3Back", features_hbr_3back)