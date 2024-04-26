import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# link: https://ajaytech.co/python-logistic-regression/

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
    selected_rows = df[(df['patient_id'] == "C1") & df['n_back'].isin([0, 3])]
    
    if not selected_rows.empty:
        selected_rows['n_back'] = selected_rows['n_back'].replace(3, 1)

        # Extract features and target from selected rows
        X_test = selected_rows.drop(columns=['n_back', 'patient_id'])
        y_test = selected_rows['n_back']
    else:
        # If there are no rows with 'patient_id' equal to "C20" and 'n_back' values 0 or 3, assign None to X_test and y_test
        X_test = None
        y_test = None


# Select rows where 'patient_id' is not equal to "C20" and 'n_back' values are 0 or 3
selected_rows = df[(df['patient_id'] != "C1") & df['n_back'].isin([0, 3])]

if not selected_rows.empty:
    selected_rows['n_back'] = selected_rows['n_back'].replace(3, 1)

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

# TRAIN
theta_power_train = X_train[:, 0]
exponent_train = X_train[:, 1]
labels_train = y_train

plt.scatter(theta_power_train, exponent_train, c=labels_train)
plt.xlabel("Theta Power")
plt.ylabel("Exponent")
plt.show()

plt.scatter(theta_power_train, labels_train)
plt.xlabel("Theta Power")
plt.ylabel("N-Back Task")
plt.show()

# TEST
theta_power_test = X_test[:, 0]
exponent_test = X_test[:, 1]
labels_test = y_test

# Logit function

from scipy.special import expit

x = np.linspace(0, 0.999, num=100)
y = np.log(x/(1-x))

plt.plot(x,y)
plt.grid()
plt.show()
 
x_new = y
y_new = expit(x_new)
 
plt.plot(x_new,y_new)
plt.grid()
plt.show()


S = StandardScaler()
X_train = S.fit_transform(theta_power_train.reshape(-1,1))
X_test = S.transform(theta_power_test.reshape(-1,1))
 
model = LogisticRegression(C=1e5, solver='lbfgs')
model.fit(X_train, labels_train)

plt.scatter(theta_power_train, labels_train)
plt.show()

x_test = np.linspace(0.1, 2, 100)
# predict dummy y_test data based on the logistic model

#x_test = x_test.reshape(1, -1)

# Now, the expression should work without broadcasting issues
#y_test = np.dot(model.coef_, x_test) 
#y_test = y_test.T + model.intercept_

y_test = x_test * model.coef_ + model.intercept_
sigmoid = expit(y_test)

plt.scatter(theta_power_train, labels_train, c=labels_train, label = "Theta power")
plt.plot(x_test, sigmoid.ravel(), c="green", label = "logistic fit")
plt.axhline(.5, color="red", label="cutoff")
plt.legend(loc="lower right")
plt.show()

"""
y_predict = model.predict(X_test)

print ( "confusion matrix = \n" , confusion_matrix(labels_test, y_predict) )
print ( "accuracy score = ", accuracy_score(labels_test, y_predict) )

# Plot logistic regression curve for the test data set

# 1. Plot the N-bacl on y-axis and Theta on x-axis
plt.scatter(theta_power_train, labels_train, c= labels_train, label = "theta power")
plt.show()

# 2. Plot the logistic regression curve based on the sigmoid function
# ravel to convert the 2-d array to a flat array
plt.plot(x_test.reshape(-1,1), sigmoid, c="green", label = "logistic fit")
 
plt.yticks([0, 0.5, 1.5, 2.5, 3, 2.5])
 
# Draw a horizontal line (in red) indicating the threshold (cutoff) probability
plt.axhline(.5, color="red", label="cutoff")
plt.axhline(1.5, color="red", label="cutoff")
plt.axhline(2.5, color="red", label="cutoff") 

# Draw a vertical line (in purple) indicating the threshold (cutoff) sepal length
plt.axvline(5.4, color="purple", label="")
 
# Use text to show the Negative and positive values
plt.text(5.50,0.9,"<--True Negative-->")
plt.text(4.3,0.9,"<--False Negative-->")
plt.text(4.4,0.05,"<--True Positive-->")
plt.text(5.5,0.05,"<--False Positive-->")

plt.show()
"""