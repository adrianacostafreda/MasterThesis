import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin


current_directory = "/Users/adriana/Documents/GitHub/thesis/MasterThesis/"

# ---- Load ----
df = pd.read_csv(os.path.join(current_directory, "eeg_features.csv"))

back0 = df[df['n_back'] == 0]
back1 = df[df['n_back'] == 1]
back2 = df[df['n_back'] == 2]
back3 = df[df['n_back'] == 3]

"""
plt.figure(figsize=(10,10))
scatter1 = plt.scatter(back0['theta frontal region'], back0['Exponent'], c='red')
scatter1.set_label('0-Back')
scatter2 = plt.scatter(back1['theta frontal region'], back2['Exponent'], c='blue')
scatter2.set_label('1-Back')
scatter3 = plt.scatter(back2['theta frontal region'], back2['Exponent'], c='green')
scatter3.set_label('2-Back')
scatter4 = plt.scatter(back3['theta frontal region'], back3['Exponent'], c='orange')
scatter4.set_label('3-Back')

# Add labels and title
plt.xlabel('Theta Frontal Region', fontsize=12)
plt.ylabel('Theta Exponent', fontsize=12)
plt.title('Scatter Plot of Theta Power and Exponent for healthy controls', fontsize=14)

# Add legend
plt.legend()
"""

scatter_matrix(df, alpha=0.5, figsize=(20, 20))

df.hist(alpha=0.5, figsize=(20, 20), color='green')
#plt.show()

subjects = []

for index, row in back0.iloc[1:].iterrows():

    id = row['patient_id']
    subjects.append(id)


if (df['patient_id'] == "C20").any():
    # Select rows where 'patient_id' is "C1" and store them in the test set
    X_test = df[df['patient_id'] == "C20"].drop(columns=['n_back', 'patient_id'])
    y_test = df[df['patient_id'] == "C20"]['n_back']

else:
    # If there are no rows with 'patient_id' equal to "C1", assign None to X_test and y_test
    X_test = None
    y_test = None

# Remove rows with 'patient_id' equal to "C1" from the dataset
df = df[df['patient_id'] != "C20"]

# The remaining data becomes the training set
X_train = df.drop(columns=['n_back', 'patient_id'])
y_train = df['n_back']

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

class Multinomial_Regression(BaseEstimator, ClassifierMixin): 
    def __init__(self, X, y, params=None):     
        if (params == None):
            self.learningRate = 0.005                  # Learning Rate
            self.max_epoch = 3000                      
        else:
            self.learningRate = params['LearningRate']
            self.max_epoch = params['Epoch'] # Epochs
           
        self.weight = np.array([[0.4, 0.3, 0.2, 0.1],
                               [0.4, 0.3, 0.2, 0.1],
                               [0.4, 0.3, 0.2, 0.1],
                               [0.4, 0.3, 0.2, 0.1],
                               [0.4, 0.3, 0.2, 0.1]])
    
    def cost_derivate_gradient(self, n, Ti, Oi, X):

        result = -(np.dot(X.T,(Ti - Oi)))/n   

        return result 
    
    def function_cost_J(self,n,Ti,Oi):

        result = -(np.sum(Ti * np.log(Oi)))/n 

        return result
    
    def one_hot_encoding(self,Y):
        OneHotEncoding = []
        encoding = []

        for i in range(len(Y)):
            if(Y[i] == 0) : encoding = np.array([1,0,0,0]) #Class 0, if y = 0
            elif(Y[i] == 1) : encoding = np.array([0,1,0,0]) #Class 1, if y = 1
            elif(Y[i] == 2) : encoding = np.array([0,0,1,0]) #Class 2, if y = 2
            elif(Y[i] == 3) : encoding = np.array([0,0,0,1]) #Class 3, if y = 3

        OneHotEncoding.append(encoding)

        return OneHotEncoding
    
    def accuracy_graphic(self, answer_graph):
        labels = 'Hits', 'Faults'
        sizes = [96.5, 3.3]
        explode = (0, 0.14)
        fig1, ax1 = plt.subplots()
        ax1.pie(answer_graph, explode=explode, colors=['green','red'], labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
        ax1.axis('equal')
        plt.show()
    
    def softmax(self,z):
        soft = (np.exp(z).T / np.sum(np.exp(z),axis=1)).T 
        return soft

    def show_probability(self, arrayProbability):
        print("Probability: [ Class 0 ,  Class 1 , Class 2 , Class 3]")
        
        arrayTotal = []
        for k in arrayProbability:

            k[0] = "%.3f" % k[0]
            k[1] = "%.3f" % k[1]
            k[2] = "%.3f" % k[2]
            k[3] = "%.3f" % k[3]
            arrayTotal.append(k)
            
        for index, data in enumerate(arrayTotal):
            prob0 = data[0] * 100
            prob1 = data[1] * 100
            prob2 = data[2] * 100
            prob3 = data[3] * 100
            string = " {}: {}%, {}%, {}%, {}%".format(index, "%.3f" % prob0, "%.3f" % prob1, "%.3f" % prob2, "%.3f" % prob3)
            print(string)
    

    def predict(self, X, y):
        acc_0back = acc_1back = acc_2back = acc_3back = 0
        v_resp = []
        n = len(y)

        Z = np.matmul(X, self.weight)

        Oi = self.softmax(Z)
        prevision = np.argmax(Oi,axis=1)

        self.show_probability(Oi)
        print("")

        procent = sum(prevision == y)/n

        print(" ID-Sample  | Class Classification |  Output |   Hoped output  ")  

        for i in range(len(prevision)):
            if(prevision[i] == 0): print(" id :",i,"          | 0 - Back       |  Output:",prevision[i],"   |",y[i])
            elif(prevision[i] == 1): print(" id :",i,"          | 1-Back   |  Output:",prevision[i],"   |",y[i])
            elif(prevision[i] == 2): print(" id :",i,"          | 2-Back     |  Output:",prevision[i],"   |",y[i])
            elif(prevision[i] == 3): print(" id :",i,"          | 3-Back     |  Output:",prevision[i],"   |",y[i])
                
        for i in range(len(prevision)):
            if((prevision[i] == y[i])and(prevision[i] == 0)) : acc_0back +=1
            elif((prevision[i] == y[i])and(prevision[i] == 1)) : acc_1back +=1
            elif((prevision[i] == y[i])and(prevision[i] == 2)) : acc_2back +=1
            elif((prevision[i] == y[i])and(prevision[i] == 3)) : acc_3back += 1
        
        correct = procent * 100
        incorrect = 100 - correct
        v_resp.append(correct)
        v_resp.append(incorrect)
        self.accuracy_graphic(v_resp)
        return "%.2f"%(correct), acc_0back, acc_1back, acc_2back, acc_3back

    def show_err_graphic(self,v_epoch,v_error):
        plt.figure(figsize=(9,4))
        plt.plot(v_epoch, v_error, "m-")
        plt.xlabel("Number of Epoch")
        plt.ylabel("Error")
        plt.title("Error Minimization")
        plt.show()

    def fit(self,X,y):
        v_epochs = []
        totalError = []
        epochCount = 0
        n = len(X)
        gradientE = []

        while(epochCount < self.max_epoch):
            Ti = self.one_hot_encoding(y)

            Z = np.matmul(X, self.weight)
            Oi = self.softmax(Z)
            erro = self.function_cost_J(n,Ti,Oi)
            gradient = self.cost_derivate_gradient(n,Ti,Oi,X)
            self.weight = self.weight - self.learningRate * gradient

            if(epochCount % 100 == 0):
                totalError.append(erro)
                gradientE.append(gradient)
                v_epochs.append(epochCount)
                print("Epoch ",epochCount," Total Error:", "%.4f" % erro)
            
            epochCount += 1
        
        self.show_err_graphic(v_epochs,totalError)

        return self
    
arguments = {'Epoch':5000, 'LearningRate':0.005}
SoftmaxRegression = Multinomial_Regression(X_train, y_train , arguments)
SoftmaxRegression.fit(X_train, y_train)

acc_test, test_0back, test_1back, test_2back, test_3back = SoftmaxRegression.predict(X_test, y_test)
print("Hits - Porcent (Test): ", acc_test,"% hits")

n_0back = 0
n_1back = 0
n_2back = 0
n_3back = 0

for i in range(len(y_test)):
    if(y_test[i] == 0):
        n_0back+=1
    elif(y_test[i] == 1):
        n_1back+=1
    elif(y_test[i] == 2):
        n_2back+=1
    elif(y_test[i] == 3):
        n_3back+=1

ac_0back = (test_0back/n_0back) * 100
ac_1back = (test_1back/n_1back) * 100
ac_2back = (test_2back/n_2back) * 100
ac_3back = (test_3back/n_3back) * 100

print("- Acurracy 0-Back :","%.2f"% ac_0back, "%")
print("- Acurracy 1-Back :","%.2f"% ac_1back, "%")
print("- Acurracy 2-Back :","%.2f"% ac_2back, "%")
print("- Acurracy 3-Back :","%.2f"% ac_3back, "%")

ig, ax = plt.subplots()

plt.bar(2.0, ac_0back,color='orange')
plt.bar(4.0, ac_1back,color='g')
plt.bar(6.0, ac_2back,color='purple')
plt.bar(8.0, ac_3back,color='purple')
plt.ylabel('Scores %')
plt.xticks([2.0, 4.0, 6.0, 8.0], ["0-Back","1-Back","2-Back", "3-Back"])
plt.title('Scores by N-Back - M.Logistic Regression')
plt.show()
        