# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 14:04:26 2013

@author: haotianhe
"""

import numpy as np
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

class NaiveBayes:

        def fit(self, X, y):

                n = X.shape[0]
                dim = X.shape[1]

                classes = sorted(set(y))

                n_classes = len(classes)

                mu = np.empty( (n_classes, dim) )
                sg = np.empty( (n_classes, dim) )
                cl_prob = np.empty( n_classes)

                for i, cl in enumerate(classes):
                        mask = (y == cl)

                        Xcl = X[mask, :]

                        mu[i, :] = Xcl.mean(axis = 0)
                        sg[i, :] = Xcl.std(axis = 0)

                        cl_prob[i] = mask.mean()

                self.mu = mu
                self.sg = sg
                self.log_cl_prob = np.log(cl_prob)
                self.classes = classes


        def predict(self, X):

                def log_normal(mu, sg, x):
                        # Gives the log probability of the

                        # Can ignore the constant factors; they come out in the
                        # normalization at the end.
                        return (- np.log(sg) - (mu - x)**2 / (2 * sg**2) ).sum()

                y_predict = np.empty( X.shape[0])

                for i in range(X.shape[0]):
                        cl_prob = np.array([
                                (log_normal(self.mu[cl,:], self.sg[cl,:], X[i,:])
                                 + self.log_cl_prob[cl])
                                for cl in range(self.mu.shape[0])])

                        y_predict[i] = self.classes[np.argmax(cl_prob)]

                return y_predict
                
###############################################################################
###############################################################################
                
def loadData(filename):
    X = np.loadtxt(filename)
    return X
    
X_train = loadData('train/X_train.txt')
y_train = loadData('train/y_train.txt')
X_test = loadData('test/X_test.txt')
y_test = loadData('test/y_test.txt')

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

GNB = NaiveBayes()
GNB.fit(X_train, y_train)
y_predict_gnb = GNB.predict(X_test)

y_pred_count_gnb = total_count(y_predict_gnb)
cmatrix_gnb = confusion_matrix(y_test, y_predict_gnb)

print "\nGaussian Naive Bayes:"
print cmatrix_gnb
print ""

recall_gnb = pre_rec(cmatrix_gnb, y_test_count)
precision_gnb = pre_rec(cmatrix_gnb, y_pred_count_gnb)
accuracy_gnb = overall_accuracy(cmatrix_gnb, y_test)

print precision_gnb
print recall_gnb
print accuracy_gnb

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