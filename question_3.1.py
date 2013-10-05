# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 18:59:29 2013

@author: haotianhe
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 14:04:26 2013

@author: haotianhe
"""

import numpy as np
from sklearn.naive_bayes import GaussianNB

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
                
def loadData(filename):
    X = np.loadtxt(filename)
    return X
    
X_train = loadData('train/X_train.txt')
y_train = loadData('train/y_train.txt')
X_test = loadData('test/X_test.txt')
y_test = loadData('test/y_test.txt')

NB = NaiveBayes()
GNB = GaussianNB()

NB.fit(X_train, y_train)
GNB.fit(X_train, y_train)

y_predict = NB.predict(X_test)
ytr_predict = NB.predict(X_train)

        
###############################################################################
###############################################################################
ercountte = 0.0
ercounttr = 0.0

for p in range(len(y_test)):
    if (y_test[p] != y_predict[p]):
        ercountte += 1

for q in range(len(y_train)):
    if (y_train[q] != ytr_predict[q]):
        ercounttr += 1
        
test_rate = ercountte / len(y_test)
train_rate = ercounttr / len(y_train)
print "The training error rate of HW7 is", train_rate
print "The test error rate of HW7 is", test_rate


y_predict_g = GNB.predict(X_test)
ytr_predict_g = GNB.predict(X_train)

ercountte_g = 0.0
ercounttr_g = 0.0

for p in range(len(y_test)):
    if (y_test[p] != y_predict_g[p]):
        ercountte_g += 1

for q in range(len(y_train)):
    if (y_train[q] != ytr_predict_g[q]):
        ercounttr_g += 1
        
test_rate_g = ercountte_g / len(y_test)
train_rate_g = ercounttr_g / len(y_train)

print "The training error rate of sklearn is", train_rate_g
print "The test error rate of sklearn is", test_rate_g