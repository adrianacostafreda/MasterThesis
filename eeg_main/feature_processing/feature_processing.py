import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# link: https://github.com/jonzamora/Epileptic-Seizure-Prediction/blob/main/src/eeg_analysis.ipynb 

current_directory = "/Users/adriana/Documents/GitHub/thesis/MasterThesis/"

# ---- Load ----
df = pd.read_csv(os.path.join(current_directory, "eeg_features.csv"))

back0 = df[df['n_back'] == 0]
# back1 = df[df['n_back'] == 1]
# back2 =  df[df['n_back'] == 2]
# back3 =  df[df['n_back'] == 3]
subjects = []

for index, row in back0.iloc[1:].iterrows():

    id = row['patient_id']
    subjects.append(id)


if (df['patient_id'] == "C1").any():
    # Select rows where 'patient_id' is "C1" and store them in the test set
    X_test = df[df['patient_id'] == "C1"].drop(columns=['n_back', 'patient_id'])
    y_test = df[df['patient_id'] == "C1"]['n_back']

else:
    # If there are no rows with 'patient_id' equal to "C1", assign None to X_test and y_test
    X_test = None
    y_test = None

# Remove rows with 'patient_id' equal to "C1" from the dataset
df = df[df['patient_id'] != "C1"]

# The remaining data becomes the training set
X_train = df.drop(columns=['n_back', 'patient_id'])
y_train = df['n_back']

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

S = StandardScaler()
X_train = S.fit_transform(X_train)
X_test = S.transform(X_test)

model = LogisticRegression(multi_class='multinomial',
                                  solver='lbfgs')
model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
train_accuracy = model.score(X_train, y_train)
y_pred_test = model.predict(X_test)
test_accuracy = model.score(X_test, y_test)
print("Training accuracy was: {}".format(train_accuracy))
print("Test accuracy was: {}".format(test_accuracy))

result = pd.DataFrame({'Actual' : y_test, 'Predicted' : y_pred_test})
print(result)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test)
print("True negative rate: {}".format(tn))
print("True positive rate: {}".format(tp))
print("False negative rate: {}".format(fn))
print("False positive rate: {}".format(fp))



