{\rtf1\ansi\ansicpg1252\cocoartf1187\cocoasubrtf390
{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
\margl1440\margr1440\vieww10800\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural

\f0\fs24 \cf0 ===================================================\
Programs for the Final Project\
+++++++++++++++++++++++++++++++++++++++++++++++++++\
Haotian He\
haotianh@u.washington.edu\
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural
\cf0 ===================================================\
\
In order to run properly, all the program files should be put into the folder UCI_HAR_Dataset, which is the project dataset folder downloaded online. For details for each program file, please read the directory below.\
===================================================\
\
- 'README.txt'\
- 'judge_value.py': chooses the best K in K-Neighbors, the best decision tree depth, and the best number of trees in the random forest model by comparing the error rates of different values.\
- 'question_3.1.py': figures out the differences between the Gaussian Naive Bayes class coded in Homework 7 and that in sklearn toolkit by debugging the code and comparing the error rates.\
- 'question_3.2.py': reports the confusion matrices for all of the classifiers' results (Gaussian Naive Bayes from Homework 7, LDA, QDA, Logistic Regression, K-Neighbors, Decision Tree, and Random Forest).\
- 'question_4.py': reports the confusion matrices for all of the classifiers and their 20-dimension and 50-dimension reduced-to results (As above).\
- 'question_4_2.py': plots the scatter for the test data reduced to 2-dimension, reports the confusion matrices for the SVM with three different kernels ('rbf', 'linear', and 'polynomial').\
- 'question_5.py': reports the overall error rates and the error rates of classifying 6 for all of the classifiers (As above).\
\
\
Notes:\
- Except 'question_5.py', all the numbers reported are per cent.}