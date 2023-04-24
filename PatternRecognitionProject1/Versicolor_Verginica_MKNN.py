# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 11:26:05 2020

@author: Sourabh Kumar
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 10:13:40 2020

@author: Sourabh Kumar
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 00:24:18 2020

@author: Sourabh Kumar
"""

from sklearn.datasets import load_iris
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier 
from sklearn import metrics
import numpy as np 
from sklearn.model_selection import train_test_split
import pandas as pd 
import matplotlib.pyplot as plt1
import matplotlib.pyplot as plt2


iris = load_iris()
iris_df=pd.DataFrame(iris.data[50:150,:],columns=iris.feature_names)
summary = iris_df.describe()
summary = summary.transpose()

print('IRIS dataset Summary-\n', summary)


# No. of Features Variations
# 1. 2 feautres - iris.data[50:150,0:2]
# 2. 3 features - iris.data[50:150,0:3]
# 3. 4 features - iris.data
X = iris.data[50:150,0:4]

#Data Normalization Variations
# 1. Non-Normalized data - X
# 2. Normalized data - preprocessing.scale(X)

X = preprocessing.scale(X)

y = iris.target[0:100]
# Splitting the data into this ratio (training:testing)=60:40
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.4,random_state=0)

k_values = range(1,11)
scores_knn = {} #scores dictionary
scores_list_knn = [] #scores list
error_cal_mean = []
 
for k in k_values:    
    # Distance Function Variations 
    # 1. L1 Distance - p=1 
    # 2. L2 Distance - p=2
    # 3. Lmax Distance - p=float('inf')
    # weights - 'distance' (mknn algo)  
    knn = KNeighborsClassifier(n_neighbors=k,weights='distance',p=float('inf'),metric='minkowski') 
    knn.fit(X_train,y_train)
    y_pred = knn.predict(X_test)
    scores_knn[k] = metrics.accuracy_score(y_test,y_pred)
    scores_list_knn.append(scores_knn[k])    
    error_cal_mean.append(np.mean(y_pred != y_test))

print('Max Accuracy rate-', max(scores_list_knn), 'and for K = ', np.argmax(scores_list_knn)+1)
print('Min Error rate-', min(error_cal_mean), 'and for K = ', np.argmin(error_cal_mean)+1)

plt1.plot(k_values,scores_list_knn)
plt1.xlabel('Value of K')
plt1.ylabel('Testing Accuracy')
plt1.show()

plt2.plot(k_values,error_cal_mean)
plt2.xlabel('Value of K')
plt2.ylabel('Error Calc using Mean')
plt2.show()





