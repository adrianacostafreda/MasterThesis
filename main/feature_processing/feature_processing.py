import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


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


# We want that X train corresponds to 88 rows with 3 arguments each 

X_train, X_test = [], []

for index, row in df.iterrows():
    
    if row["patient_id"] == "C1":
        temp_test = []
    
        temp_test.append(row['theta frontal region'])
        temp_test.append(row['theta central region'])
        temp_test.append(row['theta parietal region'])
        temp_test.append(row['theta occipital region'])
        temp_test.append(row['theta left temporal'])
        temp_test.append(row['theta right temporal'])
        temp_test.append(row["Exponent"])
        
        X_test.append(temp_test)
    
    else:
        temp_train = []
    
        temp_train.append(row['theta frontal region'])
        temp_train.append(row['theta central region'])
        temp_train.append(row['theta parietal region'])
        temp_train.append(row['theta occipital region'])
        temp_train.append(row['theta left temporal'])
        temp_train.append(row['theta right temporal'])
        temp_train.append(row["Exponent"])
    
        X_train.append(temp_train)


X_test = np.array(X_test)
X_train = np.array(X_train)

print(X_train.shape)
print(X_test.shape)

# Convert lists to arrays after the loop
#X_train = np.vstack(X_train).T  # Transpose to ensure features are in columns
#X_test = np.vstack(X_test).T    # Transpose to ensure features are in columns

# Sample labels: 0 back, 1 back, 2 back, and 3 back
y = np.array([0, 1, 2, 3])

# Repeat the labels to match the number of samples in X_train and X_test
y_train = np.repeat(y, X_train.shape[0] // len(y))
y_test = np.repeat(y, X_test.shape[0] // len(y))

# As a final step, we'll resacle each of our features so that they
# have a mean of zero and a standard deviation of one. Again, the
# reason why we do this is a bit technical, but in general ML model
# don't like it when the features are on different scales.

S = StandardScaler()
X_train = S.fit_transform(X_train)
X_test = S.transform(X_test)

model = LogisticRegression(C=0.05, multi_class='multinomial',random_state=0)
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

print(np.mean(y_train))
