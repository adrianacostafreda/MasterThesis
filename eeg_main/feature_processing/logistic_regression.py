import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# link: https://www.geeksforgeeks.org/plot-multinomial-and-one-vs-rest-logistic-regression-in-scikit-learn/ 

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
    # Select rows where 'patient_id' is "C20" and 'n_back' values are 0 or 3
    selected_rows = df[(df['patient_id'] == "C1")]
    
    if not selected_rows.empty:

        # Extract features and target from selected rows
        X_test = selected_rows.drop(columns=['n_back', 'patient_id'])
        y_test = selected_rows['n_back']
    else:
        # If there are no rows with 'patient_id' equal to "C20" and 'n_back' values 0 or 3, assign None to X_test and y_test
        X_test = None
        y_test = None

print(y_test)
print(X_test.shape)

# Select rows where 'patient_id' is not equal to "C20" and 'n_back' values are 0 or 3
selected_rows = df[(df['patient_id'] != "C1")]

if not selected_rows.empty:

    # Extract features and target from selected rows
    X_train = selected_rows.drop(columns=['n_back', 'patient_id'])
    y_train = selected_rows['n_back']
else:
    # If there are no such rows, assign None to X_train and y_train
    X_train = None
    y_train = None

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)


"""

Use only one feature --> theta power 

"""

S = StandardScaler()
X_train = S.fit_transform(X_train)
X_test = S.transform(X_test)
 
model = LogisticRegression(multi_class='multinomial',
                                  solver='lbfgs')
model.fit(X_train, y_train)

y_pred_multi = model.predict(X_test)

# evaluate the performance of the models 
# using accuracy score and confusion matrix
print('Multinomial logistic regression accuracy:',
      accuracy_score(y_test, y_pred_multi))

conf_mat_multi = confusion_matrix(y_test, y_pred_multi)

# plot the confusion matrices
fig, axs = plt.subplots(figsize=(10, 5))
axs.imshow(conf_mat_multi, cmap=plt.cm.Blues)
axs.set_title('Multinomial logistic regression')
axs.set_xlabel('Predicted labels')
axs.set_ylabel('True labels')
axs.set_xticks(np.arange(len(y_test)))
axs.set_xticklabels(y_test)
axs.set_yticks(np.arange(len(y_test)))
axs.set_yticklabels(y_test)
plt.show()

# Plot the decision boundaries
x_min, x_max = X_train[:, 0].min() - .5, X_train[:, 0].max() + .5
y_min, y_max = X_train[:, 1].min() - .5, X_train[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, .02),
                     np.arange(y_min, y_max, .02))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure(1, figsize=(4, 3))
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolors='k',
            cmap=plt.cm.Paired)
plt.xlabel('Theta power')
plt.ylabel('Delta power')
plt.show()
