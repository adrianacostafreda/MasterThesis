import math
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold


class DataLoader_v3():
    """
    """
    def __init__(self, healthy_data_path: str, healthy_label_path: str, doc_data_path: str, icu_data_path: str) -> None:
        """
        """
        if (healthy_data_path == "" or healthy_data_path is None) or (healthy_label_path == "" or healthy_label_path is None) or (doc_data_path == "" or doc_data_path is None):
            print("Specify the path of the raw data and the labels.")
            return None
        
        self.healthy_data = np.load(healthy_data_path) # patients X channels X samples
        self.healthy_events = np.load(healthy_label_path) # patients X 8 events

        self.start = 76
        self.end =  230

        # Calculate the feature matrix for the healthy controls
        self.healthy_features = None 
        for id,patient in enumerate(self.healthy_data):
            events = self.healthy_events[id]
            for iid, _ in enumerate(events):
                event_start = iid * 153
                if iid % 2 == 0:
                    mean = np.expand_dims(np.mean(patient[:,event_start+self.start:event_start+self.end],axis=1), axis=1)
                    slope_1 = np.expand_dims(self.slopePerChannel(patient[:,event_start:event_start+153].T), axis=1)
                    slope_2 = np.expand_dims(self.slopePerChannel(patient[:,event_start+154:event_start+2*153].T), axis=1)
                    features =np.expand_dims(np.concatenate((mean, slope_1, slope_2), axis=1), axis=0)

                    if iid == 0:
                        features_per_event = features
                    else:
                        features_per_event = np.concatenate((features_per_event, features),  axis=0)

            if id == 0:
                self.healthy_features = np.expand_dims(np.mean(features_per_event, axis=0), axis=0)
            else:
                self.healthy_features = np.concatenate((self.healthy_features, np.expand_dims(np.mean(features_per_event, axis=0), axis=0)),  axis=0)

        self.icu_data = np.load(icu_data_path) # patients X channels X samples
        print(self.icu_data.shape)
        # Calculate the feature matrix for the DoC patients
        self.icu_features = None 
        for id,patient in enumerate(self.icu_data):
            for iid, _ in enumerate(events):
                event_start = iid * 153
                if iid % 2 == 0:
                    mean = np.expand_dims(np.mean(patient[:,event_start+self.start:event_start+self.end],axis=1), axis=1)
                    slope_1 = np.expand_dims(self.slopePerChannel(patient[:,event_start:event_start+153].T), axis=1)
                    slope_2 = np.expand_dims(self.slopePerChannel(patient[:,event_start+154:event_start+2*153].T), axis=1)
                    features =np.expand_dims(np.concatenate((mean, slope_1, slope_2), axis=1), axis=0)

                if iid == 0:
                    features_per_event = features
                else:
                    features_per_event = np.concatenate((features_per_event, features),  axis=0)

            if id == 0:
                self.icu_features = np.expand_dims(np.mean(features_per_event, axis=0), axis=0)
            else:
                self.icu_features = np.concatenate((self.icu_features, np.expand_dims(np.mean(features_per_event, axis=0), axis=0)),  axis=0)


        self.doc_data = np.load(doc_data_path) # patients X channels X samples
        # Calculate the feature matrix for the DoC patients
        self.doc_features = None 
        for id,patient in enumerate(self.doc_data):
            for iid in range(24):
                event_start = iid * 153 * 2 
                mean = np.expand_dims(np.mean(patient[:,event_start+self.start:event_start+self.end],axis=1), axis=1)
                slope_1 = np.expand_dims(self.slopePerChannel(patient[:,event_start:event_start+153].T), axis=1)
                slope_2 = np.expand_dims(self.slopePerChannel(patient[:,event_start+154:event_start+2*153].T), axis=1)
                features =np.expand_dims(np.concatenate((mean, slope_1, slope_2), axis=1), axis=0)

                if iid == 0:
                    features_per_event = features
                else:
                    features_per_event = np.concatenate((features_per_event, features),  axis=0)

            if id == 0:
                self.doc_features = np.expand_dims(np.mean(features_per_event, axis=0), axis=0)
            else:
                self.doc_features = np.concatenate((self.doc_features, np.expand_dims(np.mean(features_per_event, axis=0), axis=0)),  axis=0)
    
    def slopePerChannel(self, data):
        '''
        data: n_channels X samples
        '''
        # Calculate slope for each event for each channel: events X channels X 1 slope
        return np.polyfit(np.arange(data.shape[0]),data,1)[0]
    
    def getFeatures(self):
        return self.healthy_features[:,[0,1,4,5],:], self.doc_features[:,[0,1,4,5],:], self.icu_features[:,[0,1,4,5],:]
        
path = "L:\\LovbeskyttetMapper\\CONNECT-ME\\DTU\\Alex_Data\\HealthyPatients\\data_initial\\"
path_icu = "L:\\LovbeskyttetMapper\\CONNECT-ME\\DTU\\Alex_Data\\ICUPatients\\data_initial\\"
path_doc = "L:\\LovbeskyttetMapper\\CONNECT-ME\\DTU\\Alex_Data\\DoC\\AllDoC\\initial\\"
data_loader = DataLoader_v3(path + "hemo.npy", path+"event_order.npy", path_doc + "hemo.npy", path_icu + "hemo.npy")
healthy_data, doc_data, icu_data = data_loader.getFeatures()
print(healthy_data.shape)


for i in range(4):
    # Creating 3d figure
    # fig = plt.figure(figsize = (10, 7))
    # ax = plt.axes(projection ="3d")
    # ax.scatter3D(X_class0[:,i,0], X_class0[:,i,1], X_class0[:,i,2], color = "green")
    # ax.scatter3D(X_class1[:,i,0], X_class1[:,i,1], X_class1[:,i,2], color = "blue")
    # plt.title("simple 3D scatter plot")

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.scatter(healthy_data[:,i,1], healthy_data[:,i,1] - healthy_data[:,i,2] , s=11, c='r')
    ax.axhline(y=0.0, xmin=0.0, xmax=1.0, color='b', linestyle='--')
    ax.axvline(x=0.0, ymin=0.0, ymax=1.0, color='b', linestyle='--')
    plt.scatter(doc_data[:,i,1], doc_data[:,i,1] - doc_data[:,i,2] , s=11, c='k')
    #plt.scatter(icu_data[:,i,1], icu_data[:,i,1] - icu_data[:,i,2] , s=11, c='g')
    plt.xlabel('mean', {'fontsize' : 12})
    plt.ylabel('slope_stimuli - slope_rest', {'fontsize' : 12})
    plt.title("Channel " + str(i+1), {'fontsize' : 12})
plt.show()

# channels = [0,1,4,5]
# data = data[:,channels,:]

# ss  = StratifiedKFold(n_splits=4, shuffle=True, random_state=7)
# ss.get_n_splits(data, labels)
# accuracy = []
# f1 = []
# for i, (train, test) in enumerate(ss.split(data, labels)):
#     print(data[test].shape)

#     X_train, y_train = data[train], labels[train]
#     X_test, y_test = data[test], labels[test]
#     print(X_train.shape)
#     print(X_test.shape)

#     # Flatten last two dimensions
#     X_train = X_train.reshape(X_train.shape[0],X_train.shape[1]*X_train.shape[2])
#     X_test = X_test.reshape(X_test.shape[0],X_test.shape[1]*X_test.shape[2])

#     # Standardize
#     sc = StandardScaler()
#     X_train = sc.fit_transform(X_train)
#     X_test = sc.transform(X_test)

#     print(X_train.shape)
#     print(X_test.shape)
#     print(y_train.shape)
#     print(y_test.shape)

#     svm_clf = svm.SVC(kernel='linear') # Linear Kernel
#     # train the model
#     svm_clf.fit(X_train, y_train)
#     # make predictions
#     svm_clf_pred = svm_clf.predict(X_test)
#     # make predictions validation
#     #svm_clf_pred_validation = svm_clf.predict(X_validation)
#     # print the accuracy
#     print("Accuracy of Support Vector Machine: ",
#         accuracy_score(y_test, svm_clf_pred))
#     # print other performance metrics
#     print("Precision of Support Vector Machine: ",
#         precision_score(y_test, svm_clf_pred, average='weighted'))
#     print("Recall of Support Vector Machine: ",
#         recall_score(y_test, svm_clf_pred, average='weighted'))
#     print("F1-Score of Support Vector Machine: ",
#         f1_score(y_test, svm_clf_pred, average='weighted'))
#     print("___________________________\n")

#     accuracy.append(accuracy_score(y_test, svm_clf_pred))
#     f1.append(f1_score(y_test, svm_clf_pred, average='weighted'))

#     cm = confusion_matrix(y_test,svm_clf_pred)
#     cm_display = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ["Class_0", "Class_1"])
#     cm_display.plot()
#     #plt.show()

# print("Accuracy for each fold: " + str(accuracy) + " mean: " + str(sum(accuracy) / len(accuracy) ))
# print("F1 for each fold: " + str(f1) + " mean: " + str(sum(f1) / len(f1) ))