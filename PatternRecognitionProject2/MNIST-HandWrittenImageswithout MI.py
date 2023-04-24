# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 01:23:56 2020

@author: Sourabh Kumar
"""

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import matplotlib.pyplot as plt_bar
import matplotlib.pyplot as plt_scatter
from sklearn.preprocessing import binarize

#Load the dataset into mnist bunch or dictionary object
mnist = fetch_openml('mnist_784', cache=False)

#binarize() - Binarizes all the data set across 784 feaures based on following logic
#range [0,127]     – Binary value 0 
#range [128,255]   – Binary value 1

binary_mnist_data = binarize(mnist.data) #binary_mnist_data.shape=(70000, 784)
binary_mnist_target = mnist.target.astype(np.int)  #binary_mnist_target.shape = (70000,)

#Case-1 :Extracting data for only Class 0 and 1 pairs
mnist_training_data_0_1 = binary_mnist_data[0:12000,:] #taking only first 12000 data from training set for class 0 & 1 - (12000, 784)
mnist_test_data_0_1 = binary_mnist_data[60000:62000,:] #taking only first 2000 data from test set for class 0 & 1      - (2000, 784)
mnist_training_target_0_1 = binary_mnist_target[0:12000,] #taking only first 12000 data from test set for class 0 & 1  - (12000,)
mnist_test_target_0_1 = binary_mnist_target[60000:62000,] #taking only first 2000 data from test set for class 0 & 1   - (2000,)

#Case-2 :Extract data for only Class 7 and 9 pairs
mnist_training_data_7_9 = np.concatenate((binary_mnist_data[42000:48000:],binary_mnist_data[54000:60000:])) #taking concat of Class 7 & Class 9 features tarining data
mnist_test_data_7_9 = np.concatenate((binary_mnist_data[67000:68000,:],binary_mnist_data[69000:70000,:]))  #taking concat of Class 7 & Class 9 features test data
mnist_training_target_7_9 = np.concatenate((binary_mnist_target[42000:48000,],binary_mnist_target[54000:60000,]))  #taking concat of Class 7 & Class 9 features training target
mnist_test_target_7_9 = np.concatenate((binary_mnist_target[67000:68000,],binary_mnist_target[69000:70000,])) #taking concat of Class 7 & Class 9 features test target 


# Four algo are used
algo_metrics = ['knn-L1-accuracyscore','knn-L2-accuracyscore','knn-Jaccard','decisiontree']

scores_dict_0_1 = {} #scores dictionary for class 0 & 1
scores_list_0_1 = [] #scores list for class 0 & 1

scores_dict_7_9 = {} #scores dictionary for class 7 & 9
scores_list_7_9 = [] #scores list for class 7 & 9

iters = range(0,4)

# Calculating accuracy together of all the algorithms one by one for both class pairs (0 & 1) and (7 & 9)
for i in iters:  
    if algo_metrics[i] == 'knn-L1-accuracyscore':
        knn = KNeighborsClassifier(n_neighbors=3,p=1) 
        knn.fit(mnist_training_data_0_1,mnist_training_target_0_1)
        mnist_pred_target_0_1 = knn.predict(mnist_test_data_0_1)
        scores = metrics.accuracy_score(mnist_test_target_0_1,mnist_pred_target_0_1)
        scores_list_0_1.append(scores)
        
        knn.fit(mnist_training_data_7_9,mnist_training_target_7_9)
        mnist_pred_target_7_9 = knn.predict(mnist_test_data_7_9)
        scores = metrics.accuracy_score(mnist_test_target_7_9,mnist_pred_target_7_9)
        scores_list_7_9.append(scores)
        
    elif algo_metrics[i] == 'knn-L2-accuracyscore':   
        knn = KNeighborsClassifier(n_neighbors=3,p=2)  
        knn.fit(mnist_training_data_0_1,mnist_training_target_0_1)
        mnist_pred_target_0_1 = knn.predict(mnist_test_data_0_1)
        scores = metrics.accuracy_score(mnist_test_target_0_1,mnist_pred_target_0_1)
        scores_list_0_1.append(scores)
        
        knn.fit(mnist_training_data_7_9,mnist_training_target_7_9)
        mnist_pred_target_7_9 = knn.predict(mnist_test_data_7_9)
        scores = metrics.accuracy_score(mnist_test_target_7_9,mnist_pred_target_7_9)
        scores_list_7_9.append(scores)
        
    elif algo_metrics[i] == 'knn-Jaccard':     
        knn = KNeighborsClassifier(n_neighbors=3,metric='jaccard')  
        knn.fit(mnist_training_data_0_1,mnist_training_target_0_1)
        mnist_pred_target_0_1 = knn.predict(mnist_test_data_0_1)
        scores = metrics.accuracy_score(mnist_test_target_0_1,mnist_pred_target_0_1)
        scores_list_0_1.append(scores)
        
        knn.fit(mnist_training_data_7_9,mnist_training_target_7_9)
        mnist_pred_target_7_9 = knn.predict(mnist_test_data_7_9)
        scores = metrics.accuracy_score(mnist_test_target_7_9,mnist_pred_target_7_9)
        scores_list_7_9.append(scores)
        
    elif algo_metrics[i] == 'decisiontree':
        knn = DecisionTreeClassifier(criterion="entropy", splitter="best")  
        knn.fit(mnist_training_data_0_1,mnist_training_target_0_1)
        mnist_pred_target_0_1 = knn.predict(mnist_test_data_0_1)
        scores = metrics.accuracy_score(mnist_test_target_0_1,mnist_pred_target_0_1)
        scores_list_0_1.append(scores)    

        knn.fit(mnist_training_data_7_9,mnist_training_target_7_9)
        mnist_pred_target_7_9 = knn.predict(mnist_test_data_7_9)
        scores = metrics.accuracy_score(mnist_test_target_7_9,mnist_pred_target_7_9)
        scores_list_7_9.append(scores)

# Storing all the values in score dictionary for class pair (0 & 1)
scores_dict_0_1['knn-L1'] = round(scores_list_0_1[0]*100)
scores_dict_0_1['knn-L2'] = round(scores_list_0_1[1]*100)
scores_dict_0_1['knn-Jaccard'] = round(scores_list_0_1[2]*100)
scores_dict_0_1['dec-tree'] = round(scores_list_0_1[3]*100)

# Storing all the values in score dictionary for class pairs (7 & 9)
scores_dict_7_9['knn-L1'] = round(scores_list_7_9[0]*100)
scores_dict_7_9['knn-L2'] = round(scores_list_7_9[1]*100)
scores_dict_7_9['knn-Jaccard'] = round(scores_list_7_9[2]*100)
scores_dict_7_9['dec-tree'] = round(scores_list_7_9[3]*100)

#Printing the accuracy values in tabular manner

print('Class 0 and Class 1 data pair')
print('Algorithms'+'\2t'+'|'+'\2t'+'Accuracy%')
print('---------------------------------')
for k , v in scores_dict_0_1.items():
    print(k+'\t'+'|'+'\t', v)

print('Class 7 and Class 9 data pair')
print('Algorithms'+'\2t'+'|'+'\2t'+'Accuracy%')
print('---------------------------------')
for k , v in scores_dict_7_9.items():
    print(k+'\t'+'|'+'\t', v)

#Bar Plots
    
labels = scores_dict_0_1.keys()
class_0_1 = scores_dict_0_1.values()
class_7_9 = scores_dict_7_9.values()
x = np.arange(len(labels))
width = 0.35
fig, ax = plt_bar.subplots()
rects1 = ax.bar(x - width/2, class_0_1, width, label='class_0_1')
rects2 = ax.bar(x + width/2, class_7_9, width, label='class_7_9')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('AccuracyRate')
ax.set_title('Bar Plot- Accuracy Rate without Mutual Information')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(loc='lower right')

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)

fig.tight_layout()

plt_bar.show()

# scatter points

x1 = scores_dict_0_1.keys()
y1 = scores_dict_0_1.values()
x2 = scores_dict_7_9.keys()
y2 = scores_dict_7_9.values()
size=500
plt_scatter.scatter(x1, y1, s=size,c="red", alpha=0.4, label = "class_0_1")
plt_scatter.scatter(x2, y2, s=size,c="black", alpha=0.6, label = "class_7_9")
plt_scatter.ylabel('AccuracyRate')
plt_scatter.title('Bubble Plot- Accuracy Rate without Mutual Information')
plt_scatter.legend(loc='lower left')
plt_scatter.show()