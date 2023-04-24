# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 00:04:02 2020

@author: Sourabh Kumar
"""


import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.preprocessing import binarize
from sklearn import metrics

import matplotlib.pyplot as plt
#Load the dataset into mnist bunch or dictionary object
mnist = fetch_openml('mnist_784', cache=False)


#binarize() - Binarizes all the data set across 784 feaures based on following logic
#range [0,127]     – Binary value 0 
#range [128,255]   – Binary value 1
binary_mnist_data = binarize(mnist.data) #binary_mnist_data.shape=(70000, 784)
binary_mnist_target = mnist.target.astype(np.int)  #binary_mnist_target.shape = (70000,)

#Case-1 :Extract data for only Class 0
mnist_training_data_0 = binary_mnist_data[0:6000,:] #taking only first 6000 data from training set for class 0  - (6000, 784)
mnist_test_data_0 = binary_mnist_data[60000:61000,:] #taking only first 1000 data from test set for class 0     - (1000, 784)
mnist_training_target_0 = binary_mnist_target[0:6000,] #taking only first 6000 data from test set for class 0   - (6000,)
mnist_test_target_0 = binary_mnist_target[60000:61000,] #taking only first 1000 data from test set for class 0  - (1000,)

#Case-2 :Extract data for only Class 1
mnist_training_data_1 = binary_mnist_data[6000:12000,:] #taking only second 6000 data from training set for class 1 -  (6000, 784)
mnist_test_data_1 = binary_mnist_data[61000:62000,:] #taking only second 1000 data from test set for class 1         - (1000, 784)
mnist_training_target_1 = binary_mnist_target[6000:12000,] #taking only second 6000 data from test set for class 1  -  (6000,)
mnist_test_target_1 = binary_mnist_target[61000:62000,] #taking only second 1000 data from test set for class 1      - (1000,)

#Case-3 :Extract data for only Class 7
mnist_training_data_7 = binary_mnist_data[42000:48000,:] #taking only eighth 6000 data from training set for class 7 - (6000, 784)
mnist_test_data_7 = binary_mnist_data[67000:68000,:] #taking only eighth 1000 data from test set for class 7      -    (1000, 784)
mnist_training_target_7 = binary_mnist_target[42000:48000,] #taking only eighth 6000 data from test set for class 7  - (6000,)
mnist_test_target_7 = binary_mnist_target[67000:68000,] #taking only eighth 1000 data from test set for class 7   -    (1000,)

#Case-4:Extract data for only Class 9
mnist_training_data_9 = binary_mnist_data[54000:60000:] #taking only tenth 6000 data from training set for class 9 - (6000, 784)
mnist_test_data_9 = binary_mnist_data[69000:70000,:] #taking only tenth 1000 data from test set for class 9      -   (1000, 784)
mnist_training_target_9 = binary_mnist_target[54000:60000,] #taking only tenth 6000 data from test set for class 9  -(6000,)
mnist_test_target_9 = binary_mnist_target[69000:70000,] #taking only tenth 1000 data from test set for class 9   -   (1000,)

#Case-5 :Extract data for only Class 0 and 1 together
mnist_training_data_0_1 = binary_mnist_data[0:12000,:] #taking only first 12000 data from training set for class 0 & 1 - (12000, 784)
mnist_test_data_0_1 = binary_mnist_data[60000:62000,:] #taking only first 2000 data from test set for class 0 & 1      - (2000, 784)
mnist_training_target_0_1 = binary_mnist_target[0:12000,] #taking only first 12000 data from test set for class 0 & 1  - (12000,)
mnist_test_target_0_1 = binary_mnist_target[60000:62000,] #taking only first 2000 data from test set for class 0 & 1   - (2000,)

#Case-6 :Extract data for only Class 7 and 9 together
mnist_training_data_7_9 = np.concatenate((binary_mnist_data[42000:48000,:],binary_mnist_data[54000:60000,:])) #taking concat of Class 7 & Class 9 features tarining data
mnist_test_data_7_9 = np.concatenate((binary_mnist_data[67000:68000,:],binary_mnist_data[69000:70000,:]))  #taking concat of Class 7 & Class 9 features test data
mnist_training_target_7_9 = np.concatenate((binary_mnist_target[42000:48000,],binary_mnist_target[54000:60000,]))  #taking concat of Class 7 & Class 9 features training target
mnist_test_target_7_9 = np.concatenate((binary_mnist_target[67000:68000,],binary_mnist_target[69000:70000,])) #taking concat of Class 7 & Class 9 features test target 


############ Score Lists for 7 & 9#########
#cl_fi_co_la - cluster first, combine later
#co_fi_cl_la - combine first, cluster later

scores_list_0_1_cl_fi_co_la = []
scores_list_7_9_cl_fi_co_la = []
scores_list_0_1_co_fi_cl_la = []
scores_list_7_9_co_fi_cl_la = []

############ K_List is a dictionary for cluster inputs as KEYS and number of patterns in respective clusters as VALUES 

def clusters (mnist_training_data,n):    
    cluster_training_data= kmeans.fit(mnist_training_data)
    new_training_label = kmeans.labels_
    new_training_data = cluster_training_data.cluster_centers_
    return new_training_data, new_training_label

def infer_cluster_labels(new_training_data, new_training_label, actual_labels):
    inferred_labels = {}
    for i in range(len(new_training_data)):
        # find index of points in cluster
        labels = []
        index = np.where(new_training_label == i)
        # append actual labels for each point in cluster
        labels.append(actual_labels[index])
    
        # determine most common label
        if len(labels[0]) == 1:
            counts = np.bincount(labels[0])
        else:
            counts = np.bincount(np.squeeze(labels))
    
        # assign the cluster to a value in the inferred_labels dictionary
        if np.argmax(counts) in inferred_labels:
            # append the new number to the existing array at this slot
            inferred_labels[np.argmax(counts)].append(i)
        else:
            # create a new array in this slot
            inferred_labels[np.argmax(counts)] = [i]
    
        #print(labels)
        #print('Cluster: {}, label: {}'.format(i, np.argmax(counts)))
    return inferred_labels

def infer_data_labels(X_labels, cluster_labels):
    # empty array of len(X)
    predicted_labels = np.zeros(len(X_labels)).astype(np.uint8)
    for i, cluster in enumerate(X_labels):
        for key, value in cluster_labels.items():
            if cluster in value:
                predicted_labels[i] = key
    return predicted_labels

K_List_1 =[100,200,300,400,500]
K_List_2 =[200,400,600,800,1000]
neighbors = 4
iters1 = range(len(K_List_1))
iters2 = range(len(K_List_2))

for i in iters1:    
    kmeans = KMeans(n_clusters=K_List_1[i], init='random')
    new_training_data_7, new_training_label_7 = clusters(mnist_training_data_7,K_List_1[i])
    new_training_data_9, new_training_label_9 = clusters(mnist_training_data_9,K_List_1[i])
    X_clusters_7 = kmeans.predict(new_training_data_7)
    X_clusters_9 = kmeans.predict(new_training_data_9)
    cluster_labels_7 = infer_cluster_labels(new_training_data_7, new_training_label_7, mnist_training_target_7)
    cluster_labels_9 = infer_cluster_labels(new_training_data_9, new_training_label_9, mnist_training_target_9)
    new_training_target_7 = infer_data_labels(X_clusters_7, cluster_labels_7)
    new_training_target_9 = infer_data_labels(X_clusters_9, cluster_labels_9)
    new_training_data_7_9_cl_fi_co_la = np.concatenate((new_training_data_7,new_training_data_9))
    new_training_target_7_9_cl_fi_co_la = np.concatenate((new_training_target_7,new_training_target_9))
    
# ####################################################################################################################
    
# #KNN Classification on 2 (a) on combined class clusters 7 & 9 
    
    knn = KNeighborsClassifier(n_neighbors=neighbors,p=2)
    knn.fit(new_training_data_7_9_cl_fi_co_la,new_training_target_7_9_cl_fi_co_la)
    y_pred = knn.predict(mnist_test_data_7_9)
    scores = metrics.accuracy_score(mnist_test_target_7_9,y_pred)
    scores_list_7_9_cl_fi_co_la.append(scores)
print('scores_list_7_9_cl_fi_co_la-',scores_list_7_9_cl_fi_co_la)

for i in iters2:    
    kmeans = KMeans(n_clusters=K_List_2[i], init='random')
    new_training_data_7_9, new_training_label_7_9 = clusters(mnist_training_data_7_9,K_List_2[i])
    X_clusters = kmeans.predict(new_training_data_7_9)
    cluster_labels = infer_cluster_labels(new_training_data_7_9, new_training_label_7_9, mnist_training_target_7_9)
    new_training_target_7_9 = infer_data_labels(X_clusters, cluster_labels)
    new_training_data_7_9_co_fi_cl_la = new_training_data_7_9
    new_training_target_7_9_co_fi_cl_la = new_training_target_7_9
####################################################################################################################
    
#KNN Classification on 2 (a) on combined class clusters 7 & 9 
    
    knn = KNeighborsClassifier(n_neighbors=neighbors,p=2)
    knn.fit(new_training_data_7_9_co_fi_cl_la,new_training_target_7_9_co_fi_cl_la)
    y_pred = knn.predict(mnist_test_data_7_9)
    scores = metrics.accuracy_score(mnist_test_target_7_9,y_pred)
    scores_list_7_9_co_fi_cl_la.append(scores)
print('scores_list_7_9_co_fi_cl_la-',scores_list_7_9_co_fi_cl_la)


#x : list of different clusters
x = [200,400,600,800,1000]

#plot4 for Class 7 & 9 pair under Kmeans

#Line1- scores_list_7_9_cl_fi_co_la : list different accuracies of class 7 & Class 9 data pair when clustering done on combined data and clusters combined later
scores_list_7_9_cl_fi_co_la= [0.7315, 0.737, 0.661, 0.7045, 0.754]

#Line2- scores_list_7_9_co_fi_cl_la : list different accuracies of class 7 & Class 9 data pair when clustering done on pre-combined data
scores_list_7_9_co_fi_cl_la= [0.917, 0.926, 0.949, 0.9415, 0.9425]

plt.plot(x, scores_list_7_9_cl_fi_co_la, label = "clustering on uncombined data")
plt.plot(x, scores_list_7_9_co_fi_cl_la, label = "clustering on combined data")

plt.xlabel('Clusters')
plt.ylabel('Accuracy')

plt.title('Line Plot- Accuracy rate for different clusters for class pairs- 7 & 9 under k-means')
plt.legend()
plt.show()