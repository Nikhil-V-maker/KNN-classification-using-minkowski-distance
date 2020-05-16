#loading libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
'exec(%matplotlib inline)'
df = pd.read_csv('data.csv')

#df
print(df.head());
print('\n');
#removing unnecessary columns
df = df.drop(['id', 'Unnamed: 32'], axis = 1)

print(df.shape);
print('\n');

print(df.dtypes);
print('\n');

#to check whether any null value is present
print(df.isnull().sum());
print('\n');

def diagnosis_value(diagnosis):
    if diagnosis == 'M':
        return 1
    else:
        return 0

df['diagnosis'] = df['diagnosis'].apply(diagnosis_value)

sns.lmplot(x = 'radius_mean', y= 'texture_mean', hue = 'diagnosis',data = df)

sns.lmplot(x='smoothness_mean', y = 'compactness_mean', data = df, hue = 'diagnosis')

#loading libraries
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import math

X = np.array(df.iloc[:,1:])
y = np.array(df['diagnosis'])
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.33, random_state = 42)

x= int(input ("Enter the power p value for Minkowskis distance "))
print('\n');

a=math.sqrt(len(X_train))
b=round(a)

knn = KNeighborsClassifier(n_neighbors=20,p=x)
knn.fit(X_train,y_train)

print(knn.score(X_test,y_test));
print('\n');

#Performing cross validation
neighbors = []
cv_scores = []
from sklearn.model_selection import cross_val_score
#perform 10 fold cross validation
# for k in range(1,51,2):
#     neighbors.append(k)
knn = KNeighborsClassifier(n_neighbors = 20)
scores = cross_val_score(knn,X_train,y_train,cv=10, scoring = 'accuracy')
print(scores.mean());
  
    

