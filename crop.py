import numpy as np  
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Crop_recommendation.csv')

#Convert to factorize the dataset
#pd.factorize() is useful for obtaining a numeric representation
#dataset['city'] = pd.factorize(dataset['city'])[0]
#print('city',dataset)

#Split the X & Y variable
X = dataset.iloc[:, [0,1,2,6]].values
y = dataset.iloc[:,7].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
#X_train = pd.get_dummies(0,1)
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting RF to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

z = classifier.predict([[2,22,39,49.9472228]])
print(z)

   
#9,15,48,11.7335312-APPLE
#100,90,40,80.3333319-COFFEE
#19,28,38,4.4477319-GRAPES





