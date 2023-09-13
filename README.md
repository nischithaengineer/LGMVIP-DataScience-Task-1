# LGMVIP-DataScience-Task-1
#  Iris Flower Classification
The iris flower classification aims to predict flower classes (Versicolor, Setosa, Virginica) using four features: Sepal length, Sepal width, Petal length, and Petal width using Support vector machine algorithm a supervised machine learning algorithms 

# Import Packages
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
%matplotlib inline

# Load the data

columns = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width', 'Class_lab
df = pd.read_csv('irisdata.csv', names=columns)
df.head()

# statistical analysis about the data
df.describe()

# Visualize the whole dataset

custom_palette = ['red', 'blue', 'green']
sns.pairplot(df, hue='Class_labels', palette=custom_palette)
plt.show()

# Separate features and target  
data = df.values
X = data[:,0:4]
Y = data[:,4]

# Calculate average of each features for all classes
Y_Data = np.array([np.average(X[:, i][Y==j].astype('float32')) for i in range (X.shape[1])
 for j in (np.unique(Y))])
Y_Data_reshaped = Y_Data.reshape(4, 3)
Y_Data_reshaped = np.swapaxes(Y_Data_reshaped, 0, 1)
X_axis = np.arange(len(columns)-1)
width = 0.25

# Plot the average
colors = ['red', 'blue', 'green']
plt.bar(X_axis, Y_Data_reshaped[0], width, label='Setosa', color=colors[0])
plt.bar(X_axis + width, Y_Data_reshaped[1], width, label='Versicolour', color=color
plt.bar(X_axis + width * 2, Y_Data_reshaped[2], width, label='Virginica', color=col
plt.xticks(X_axis, columns[:4])
plt.xlabel("Features")
plt.ylabel("Value in cm.")
plt.legend(bbox_to_anchor=(1.3, 1))
plt.show()

# Split the data to train and test dataset.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# Support vector machine algorithm
from sklearn.svm import SVC
svn = SVC()
svn.fit(X_train, y_train)

# Predict from the test dataset
predictions = svn.predict(X_test)

# Calculate the accuracy
from sklearn.metrics import accuracy_score
accuracy_score(y_test, predictions)

# A detailed classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))

# Prediction of the species from the input vector
X_new = np.array([[3, 2, 1, 0.2], [  4.9, 2.2, 3.8, 1.1 ], [  5.3, 2.5, 4.6, 1.9 ]])
prediction = svn.predict(X_new)
print("Prediction of Species: {}".format(prediction))

# Save the model
import pickle
with open('SVM.pickle', 'wb') as f:
    pickle.dump(svn, f)

# Load the model
with open('SVM.pickle', 'rb') as f:
    model = pickle.load(f)
model.predict(X_new)
