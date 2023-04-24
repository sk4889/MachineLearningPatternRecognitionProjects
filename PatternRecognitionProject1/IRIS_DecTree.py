# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 03:50:40 2020

@author: Sourabh Kumar
"""

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

iris = load_iris()
X = iris.data

#Data Normalization Variations
# 1. Non-Normalized data - X
# 2. Normalized data - preprocessing.scale(X)
#X = preprocessing.scale(X)
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6, random_state=1)

# 1. splitter : {"best","random"}
# 2. criterion : {"gini","entropy"}
# 3. max_depth : {"int","none"}

dtc = DecisionTreeClassifier(criterion="entropy", splitter="random")
dtc.fit(X_train, y_train)
print(dtc.score(X_test, y_test))
