# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 15:42:04 2013

@author: haotianhe
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def loadClassificationData(filename):
    X = np.loadtxt(filename)
    return X
    
X_train = loadClassificationData('train/x_train.txt')
y_train = loadClassificationData('train/y_train.txt')
X_test = loadClassificationData('test/x_test.txt')
y_test = loadClassificationData('test/y_test.txt')

###############################################################################
# 1 K-Neighbors Classifier
for i in range(20):
    KNN = KNeighborsClassifier(i+1)
    KNN.fit(X_train, y_train)
    y_predict_kn = KNN.predict(X_test)
    
    ercountte = 0.0
    
    for p in range(len(y_test)):
        if (y_test[p] != y_predict_kn[p]):
            ercountte += 1
        
    test_rate_kn = ercountte / len(y_test)
    print "The test error rate of", i, ' for K-Neighbors Classifier is', test_rate_kn
print ""


###############################################################################
# 2 Decision Tree Classifier
for i in range(20):
    DTC = DecisionTreeClassifier(i+1)
    DTC.fit(X_train, y_train)
    y_predict_dt = DTC.predict(X_test)
    
    ercountte = 0.0
    
    for p in range(len(y_test)):
        if (y_test[p] != y_predict_dt[p]):
            ercountte += 1
        
    test_rate_dt = ercountte / len(y_test)
    print "The test error rate of", i, ' for Decision Tree Classifier is', test_rate_dt
print ""
  
  
###############################################################################
# 3 K-Random Forest Classifier
for i in range(20):
    RFC = RandomForestClassifier(i+1)
    RFC.fit(X_train, y_train)
    y_predict_rf = RFC.predict(X_test)
    
    ercountte = 0.0
    
    for p in range(len(y_test)):
        if (y_test[p] != y_predict_rf[p]):
            ercountte += 1
        
    test_rate_rf = ercountte / len(y_test)
    print "The test error rate of", i, ' for K-Random Forest Classifier is', test_rate_rf
print ""