# -*- coding: utf-8 -*-
"""
Created on Mon May 11 00:34:07 2020

@author: Sourabh Kumar
"""

import matplotlib.pyplot as plt

# line 1 plot
#x1 : list of different features
#y1 : list different accuracies of class 0 & Class 1 data pair

x1 = [80,160,240,280,290,300,320,784]
y1 = [67.0,70.0,73.0,74.0,73.0,73.0,69.0,52.0]

# line 2 plot
#x2 : list of different features
#y2 : list different accuracies of class 7 & Class 9 data pair

x2 = [80,160,240,280,290,300,320,784]
y2 = [77.0,81.0,81.0,81.0,81.0,80.0,79.0,60.0]

plt.plot(x1, y1, label = "Class-0 & 1")
plt.plot(x2, y2, label = "Class-7 & 9")

plt.xlabel('Features')
plt.ylabel('Accuracy')

plt.title('Line Plot- Accuracy rate for different features for different class pairs')
plt.legend()
plt.show()


