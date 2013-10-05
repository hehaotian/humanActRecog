# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 15:34:25 2013

@author: haotianhe
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

###############################################################################
###############################################################################

def loadData(filename):
    X = np.loadtxt(filename)
    return X
    
X_train = loadData('train/X_train.txt')
y_train = loadData('train/y_train.txt')
X_test = loadData('test/X_test.txt')
y_test = loadData('test/y_test.txt')

###############################################################################
###############################################################################

pca20 = PCA(n_components = 20)
pca20.fit(X_train)
X_train_20 = pca20.transform(X_train)
X_test_20 = pca20.transform(X_test)

pca50 = PCA(n_components = 50)
pca50.fit(X_train)
X_train_50 = pca50.transform(X_train)
X_test_50 = pca50.transform(X_test)

###############################################################################
###############################################################################

def total_count(y):
    count = np.zeros(6)
    for i in range(y.size):
        if (y[i] == 1):
            count[0] += 1
        if (y[i] == 2):
            count[1] += 1
        if (y[i] == 3):
            count[2] += 1
        if (y[i] == 4):
            count[3] += 1
        if (y[i] == 5):
            count[4] += 1
        if (y[i] == 6):
            count[5] += 1
    return count

y_test_count = total_count(y_test)

def confusion_matrix(y_test, y_pred):
    cm = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            for k in range(y_pred.size):
                if (y_pred[k] == j + 1 and y_test[k] == i + 1):
                    cm[i][j] += 1
    return cm


def pre_rec(cm, count):
    perc = []
    for i in range(6):
        perc.append(float("{0:.2f}".format(cm[i][i] / count[i] * 100)))
    return perc
        
def overall_accuracy(cm, y_test):
    sum = 0
    for i in range(6):
        sum += cm[i][i]
    return float("{0:.2f}".format(sum * 100.0 / y_test.size))    
        
###############################################################################
###############################################################################
# 1 Gaussian Naive Bayes
        
GNB = GaussianNB()
GNB.fit(X_train, y_train)
y_predict_gnb = GNB.predict(X_test)

y_pred_count_gnb = total_count(y_predict_gnb)
cmatrix_gnb = confusion_matrix(y_test, y_predict_gnb)

print "\nGaussian Naive Bayes without Dimension Reduction:"
print cmatrix_gnb
print ""

recall_gnb = pre_rec(cmatrix_gnb, y_test_count)
precision_gnb = pre_rec(cmatrix_gnb, y_pred_count_gnb)
accuracy_gnb = overall_accuracy(cmatrix_gnb, y_test)

print precision_gnb
print recall_gnb
print accuracy_gnb
        
###############################################################################

GNB = GaussianNB()
GNB.fit(X_train_20, y_train)
y_predict_gnb_20 = GNB.predict(X_test_20)

y_pred_count_gnb_20 = total_count(y_predict_gnb_20)
cmatrix_gnb_20 = confusion_matrix(y_test, y_predict_gnb_20)

print "\nGaussian Naive Bayes with Dimension Reduced to 20:"
print cmatrix_gnb_20
print ""

recall_gnb_20 = pre_rec(cmatrix_gnb_20, y_test_count)
precision_gnb_20 = pre_rec(cmatrix_gnb_20, y_pred_count_gnb_20)
accuracy_gnb_20 = overall_accuracy(cmatrix_gnb_20, y_test)

print precision_gnb_20
print recall_gnb_20
print accuracy_gnb_20

###############################################################################

GNB = GaussianNB()
GNB.fit(X_train_50, y_train)
y_predict_gnb_50 = GNB.predict(X_test_50)

y_pred_count_gnb_50 = total_count(y_predict_gnb_50)
cmatrix_gnb_50 = confusion_matrix(y_test, y_predict_gnb_50)

print "\nGaussian Naive Bayes with Dimension Reduced to 50:"
print cmatrix_gnb_50
print ""

recall_gnb_50 = pre_rec(cmatrix_gnb_50, y_test_count)
precision_gnb_50 = pre_rec(cmatrix_gnb_50, y_pred_count_gnb_50)
accuracy_gnb_50 = overall_accuracy(cmatrix_gnb_50, y_test)

print precision_gnb_50
print recall_gnb_50
print accuracy_gnb_50
print "####################################################################"

###############################################################################
###############################################################################
# 2 LDA

lda = LDA()
lda.fit(X_train, y_train)
y_predict_lda = lda.predict(X_test)

y_pred_count_lda = total_count(y_predict_lda)
cmatrix_lda = confusion_matrix(y_test, y_predict_lda)

print "\nLDA:"
print cmatrix_lda
print ""

recall_lda = pre_rec(cmatrix_lda, y_test_count)
precision_lda = pre_rec(cmatrix_lda, y_pred_count_lda)
accuracy_lda = overall_accuracy(cmatrix_lda, y_test)

print precision_lda
print recall_lda
print accuracy_lda

###############################################################################

lda = LDA()
lda.fit(X_train_20, y_train)
y_predict_lda_20 = lda.predict(X_test_20)

y_pred_count_lda_20 = total_count(y_predict_lda_20)
cmatrix_lda_20 = confusion_matrix(y_test, y_predict_lda_20)

print "\nLDA with Dimension Reduced to 20:"
print cmatrix_lda_20
print ""

recall_lda_20 = pre_rec(cmatrix_lda_20, y_test_count)
precision_lda_20 = pre_rec(cmatrix_lda_20, y_pred_count_lda_20)
accuracy_lda_20 = overall_accuracy(cmatrix_lda_20, y_test)

print precision_lda_20
print recall_lda_20
print accuracy_lda_20

###############################################################################
  
lda = LDA()
lda.fit(X_train_50, y_train)
y_predict_lda_50 = lda.predict(X_test_50)

y_pred_count_lda_50 = total_count(y_predict_lda_50)
cmatrix_lda_50 = confusion_matrix(y_test, y_predict_lda_50)

print "\nLDA with Dimension Reduced to 50:"
print cmatrix_lda_50
print ""

recall_lda_50 = pre_rec(cmatrix_lda_50, y_test_count)
precision_lda_50 = pre_rec(cmatrix_lda_50, y_pred_count_lda_50)
accuracy_lda_50 = overall_accuracy(cmatrix_lda_50, y_test)

print precision_lda_50
print recall_lda_50
print accuracy_lda_50
print "####################################################################"
                
###############################################################################
###############################################################################
# 3 QDA

qda = QDA()
qda.fit(X_train, y_train)
y_predict_qda = qda.predict(X_test)

y_pred_count_qda = total_count(y_predict_qda)
cmatrix_qda = confusion_matrix(y_test, y_predict_qda)

print "\nQDA:"
print cmatrix_qda
print ""

recall_qda = pre_rec(cmatrix_qda, y_test_count)
precision_qda = pre_rec(cmatrix_qda, y_pred_count_qda)
accuracy_qda = overall_accuracy(cmatrix_qda, y_test)

print precision_qda
print recall_qda
print accuracy_qda

###############################################################################
 
qda = QDA()
qda.fit(X_train_20, y_train)
y_predict_qda_20 = qda.predict(X_test_20)

y_pred_count_qda_20 = total_count(y_predict_qda_20)
cmatrix_qda_20 = confusion_matrix(y_test, y_predict_qda_20)

print "\nQDA with Dimension Reduced to 20:"
print cmatrix_qda_20
print ""

recall_qda_20 = pre_rec(cmatrix_qda_20, y_test_count)
precision_qda_20 = pre_rec(cmatrix_qda_20, y_pred_count_qda_20)
accuracy_qda_20 = overall_accuracy(cmatrix_qda_20, y_test)

print precision_qda_20
print recall_qda_20
print accuracy_qda_20

###############################################################################
 
qda = QDA()
qda.fit(X_train_50, y_train)
y_predict_qda_50 = qda.predict(X_test_50)

y_pred_count_qda_50 = total_count(y_predict_qda_50)
cmatrix_qda_50 = confusion_matrix(y_test, y_predict_qda_50)

print "\nQDA with Dimension Reduced to 50:"
print cmatrix_qda_50
print ""

recall_qda_50 = pre_rec(cmatrix_qda_50, y_test_count)
precision_qda_50 = pre_rec(cmatrix_qda_50, y_pred_count_qda_50)
accuracy_qda_50 = overall_accuracy(cmatrix_qda_50, y_test)

print precision_qda_50
print recall_qda_50
print accuracy_qda_50
print "####################################################################"

###############################################################################
###############################################################################
# 4 Logistic Regression

logr = LogisticRegression()
logr.fit(X_train, y_train)
y_predict_logr = logr.predict(X_test)

y_pred_count_logr = total_count(y_predict_logr)
cmatrix_logr = confusion_matrix(y_test, y_predict_logr)

print "\nLogistic Regression:"
print cmatrix_logr
print ""

recall_logr = pre_rec(cmatrix_logr, y_test_count)
precision_logr = pre_rec(cmatrix_logr, y_pred_count_logr)
accuracy_logr = overall_accuracy(cmatrix_logr, y_test)

print precision_logr
print recall_logr
print accuracy_logr

###############################################################################
 
logr = LogisticRegression()
logr.fit(X_train_20, y_train)
y_predict_logr_20 = logr.predict(X_test_20)

y_pred_count_logr_20 = total_count(y_predict_logr_20)
cmatrix_logr_20 = confusion_matrix(y_test, y_predict_logr_20)

print "\nLogistic Regression with Dimension Reduced to 20:"
print cmatrix_logr_20
print ""

recall_logr_20 = pre_rec(cmatrix_logr_20, y_test_count)
precision_logr_20 = pre_rec(cmatrix_logr_20, y_pred_count_logr_20)
accuracy_logr_20 = overall_accuracy(cmatrix_logr_20, y_test)

print precision_logr_20
print recall_logr_20
print accuracy_logr_20

###############################################################################
 
logr = LogisticRegression()
logr.fit(X_train_50, y_train)
y_predict_logr_50 = logr.predict(X_test_50)

y_pred_count_logr_50 = total_count(y_predict_logr_50)
cmatrix_logr_50 = confusion_matrix(y_test, y_predict_logr_50)

print "\nLogistic Regression with Dimension Reduced to 50:"
print cmatrix_logr_50
print ""

recall_logr_50 = pre_rec(cmatrix_logr_50, y_test_count)
precision_logr_50 = pre_rec(cmatrix_logr_50, y_pred_count_logr_50)
accuracy_logr_50 = overall_accuracy(cmatrix_logr_50, y_test)

print precision_logr_50
print recall_logr_50
print accuracy_logr_50
print "####################################################################"

###############################################################################
###############################################################################
# 5 K-Neighbors Classifier

knc = KNeighborsClassifier(8)
knc.fit(X_train, y_train)
y_predict_knc = knc.predict(X_test)

y_pred_count_knc = total_count(y_predict_knc)
cmatrix_knc = confusion_matrix(y_test, y_predict_knc)

print "\nK-Neighbors Classifier:"
print cmatrix_knc
print ""

recall_knc = pre_rec(cmatrix_knc, y_test_count)
precision_knc = pre_rec(cmatrix_knc, y_pred_count_knc)
accuracy_knc = overall_accuracy(cmatrix_knc, y_test)

print precision_knc
print recall_knc
print accuracy_knc

###############################################################################
 
knc = KNeighborsClassifier(8)
knc.fit(X_train_20, y_train)
y_predict_knc_20 = knc.predict(X_test_20)

y_pred_count_knc_20 = total_count(y_predict_knc_20)
cmatrix_knc_20 = confusion_matrix(y_test, y_predict_knc_20)

print "\nK-Neighbors Classifier with Dimension Reduced to 20:"
print cmatrix_knc_20
print ""

recall_knc_20 = pre_rec(cmatrix_knc_20, y_test_count)
precision_knc_20 = pre_rec(cmatrix_knc_20, y_pred_count_knc_20)
accuracy_knc_20 = overall_accuracy(cmatrix_knc_20, y_test)

print precision_knc_20
print recall_knc_20
print accuracy_knc_20

###############################################################################
 
knc = KNeighborsClassifier(8)
knc.fit(X_train_50, y_train)
y_predict_knc_50 = knc.predict(X_test_50)

y_pred_count_knc_50 = total_count(y_predict_knc_50)
cmatrix_knc_50 = confusion_matrix(y_test, y_predict_knc_50)

print "\nK-Neighbors Classifier with Dimension Reduced to 50:"
print cmatrix_knc_50
print ""

recall_knc_50 = pre_rec(cmatrix_knc_50, y_test_count)
precision_knc_50 = pre_rec(cmatrix_knc_50, y_pred_count_knc_50)
accuracy_knc_50 = overall_accuracy(cmatrix_knc_50, y_test)

print precision_knc_50
print recall_knc_50
print accuracy_knc_50
print "####################################################################"

###############################################################################
###############################################################################
# 6 Decision Tree Classifier

dt = DecisionTreeClassifier(max_depth = 9)
dt.fit(X_train, y_train)
y_predict_dt = dt.predict(X_test)

y_pred_count_dt = total_count(y_predict_dt)
cmatrix_dt = confusion_matrix(y_test, y_predict_dt)

print "\nDecision Tree Classifier:"
print cmatrix_dt
print ""

recall_dt = pre_rec(cmatrix_dt, y_test_count)
precision_dt = pre_rec(cmatrix_dt, y_pred_count_dt)
accuracy_dt = overall_accuracy(cmatrix_dt, y_test)

print precision_dt
print recall_dt
print accuracy_dt

###############################################################################

dt = DecisionTreeClassifier(max_depth = 9)
dt.fit(X_train_20, y_train)
y_predict_dt_20 = dt.predict(X_test_20)

y_pred_count_dt_20 = total_count(y_predict_dt_20)
cmatrix_dt_20 = confusion_matrix(y_test, y_predict_dt_20)

print "\nDecision Tree Classifier with Dimension Reduced to 20:"
print cmatrix_dt_20
print ""

recall_dt_20 = pre_rec(cmatrix_dt_20, y_test_count)
precision_dt_20 = pre_rec(cmatrix_dt_20, y_pred_count_dt_20)
accuracy_dt_20 = overall_accuracy(cmatrix_dt_20, y_test)

print precision_dt_20
print recall_dt_20
print accuracy_dt_20

###############################################################################

dt = DecisionTreeClassifier(max_depth = 9)
dt.fit(X_train_50, y_train)
y_predict_dt_50 = dt.predict(X_test_50)

y_pred_count_dt_50 = total_count(y_predict_dt_50)
cmatrix_dt_50 = confusion_matrix(y_test, y_predict_dt_50)

print "\nDecision Tree Classifier with Dimension Reduced to 50:"
print cmatrix_dt_50
print ""

recall_dt_50 = pre_rec(cmatrix_dt_50, y_test_count)
precision_dt_50 = pre_rec(cmatrix_dt_50, y_pred_count_dt_50)
accuracy_dt_50 = overall_accuracy(cmatrix_dt_50, y_test)

print precision_dt_50
print recall_dt_50
print accuracy_dt_50
print "####################################################################"

###############################################################################
###############################################################################
# 7 Random Forest Classifier

rfc = RandomForestClassifier(13)
rfc.fit(X_train, y_train)
y_predict_rfc = rfc.predict(X_test)

y_pred_count_rfc = total_count(y_predict_rfc)
cmatrix_rfc = confusion_matrix(y_test, y_predict_rfc)

print "\nRandom Forest Classifier:"
print cmatrix_rfc
print ""

recall_rfc = pre_rec(cmatrix_rfc, y_test_count)
precision_rfc = pre_rec(cmatrix_rfc, y_pred_count_rfc)
accuracy_rfc = overall_accuracy(cmatrix_rfc, y_test)

print precision_rfc
print recall_rfc
print accuracy_rfc

###############################################################################

rfc = RandomForestClassifier(13)
rfc.fit(X_train_20, y_train)
y_predict_rfc_20 = rfc.predict(X_test_20)

y_pred_count_rfc_20 = total_count(y_predict_rfc_20)
cmatrix_rfc_20 = confusion_matrix(y_test, y_predict_rfc_20)

print "\nRandom Forest Classifier with Dimension Reduced to 20:"
print cmatrix_rfc_20
print ""

recall_rfc_20 = pre_rec(cmatrix_rfc_20, y_test_count)
precision_rfc_20 = pre_rec(cmatrix_rfc_20, y_pred_count_rfc_20)
accuracy_rfc_20 = overall_accuracy(cmatrix_rfc_20, y_test)

print precision_rfc_20
print recall_rfc_20
print accuracy_rfc_20

###############################################################################

rfc = RandomForestClassifier(13)
rfc.fit(X_train_50, y_train)
y_predict_rfc_50 = rfc.predict(X_test_50)

y_pred_count_rfc_50 = total_count(y_predict_rfc_50)
cmatrix_rfc_50 = confusion_matrix(y_test, y_predict_rfc_50)

print "\nRandom Forest Classifier with Dimension Reduced to 50:"
print cmatrix_rfc_50
print ""

recall_rfc_50 = pre_rec(cmatrix_rfc_50, y_test_count)
precision_rfc_50 = pre_rec(cmatrix_rfc_50, y_pred_count_rfc_50)
accuracy_rfc_50 = overall_accuracy(cmatrix_rfc_50, y_test)

print precision_rfc_50
print recall_rfc_50
print accuracy_rfc_50