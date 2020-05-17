# -*- coding: utf-8 -*-
"""
Created on Sun May 17 23:43:48 2020

@author: megva
"""


import numpy as np;  # importing numerical computing package
from urllib.request import urlopen;  # importing url handling
from sklearn import model_selection as ms; # importing model selection tools
from sklearn import linear_model as lm; #  importing linear models
from sklearn import naive_bayes as nb; #  importing naive Bayes classifiers
from sklearn import metrics; #  importing performance metrics
from matplotlib import pyplot as plt;  # importing MATLAB-like plotting framework
import itertools;
import sklearn.preprocessing as pre;

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Greens):
    """
    Ez a funkció kinyomtatja és ábrázolja a zavart mátrixot.
    A normalizálás a `normalizálás = igaz 'beállítással alkalmazható`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalizált confuzios mátrix")
    else:
        print('Confusion matrix, normalizácio nélkül')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('Igaz címke')
    plt.xlabel('Jósolt címke')
    plt.tight_layout()
    
# Reading the dataset

url = 'https://raw.githubusercontent.com/megvagyhadnagy01/geptanulas/master/ticdata2000.txt';
label_row_number = 1;
attribute_cols = 33;
output_variable_col = 3;



raw_data = urlopen(url);
data = np.loadtxt(raw_data, delimiter="	", dtype = int, skiprows=label_row_number)
raw_data = urlopen(url);
attribute_names = np.loadtxt(raw_data, delimiter="	", dtype=str, max_rows=1)
del raw_data;



# Defining input and target variables
X = data[:,0:attribute_cols];  
y = data[:,output_variable_col];
del data;
input_names = attribute_names[0:attribute_cols];
target_names = list(range(1,3)); #rating between 1 and 10

# Partitioning into training and test sets
X_train, X_test, y_train, y_test = ms.train_test_split(X,y, test_size=0.3, 
                                shuffle = True, random_state=2020);



# Fitting logistic regression
logreg_classifier = lm.LogisticRegression();
logreg_classifier.fit(X_train,y_train);
ypred_logreg = logreg_classifier.predict(X_train);
cm_logreg_train = metrics.confusion_matrix(y_train, ypred_logreg, labels=target_names); # train confusion matrix
ypred_logreg = logreg_classifier.predict(X_test);
cm_logreg_test = metrics.confusion_matrix(y_test, ypred_logreg, labels=target_names); # test confusion matrix
yprobab_logreg = logreg_classifier.predict_proba(X_test);  #  prediction probabilities



# Fitting naive Bayes classifier
naive_bayes_classifier = nb.GaussianNB();
naive_bayes_classifier.fit(X_train,y_train);
ypred_naive_bayes = naive_bayes_classifier.predict(X_train);  # prediction for train
cm_naive_bayes_train = metrics.confusion_matrix(y_train, ypred_naive_bayes, labels=target_names); # train confusion matrix
ypred_naive_bayes = naive_bayes_classifier.predict(X_test);  #prediction
cm_naive_bayes_test = metrics.confusion_matrix(y_test, ypred_naive_bayes,labels=target_names); # test confusion matrix 
yprobab_naive_bayes = naive_bayes_classifier.predict_proba(X_test);  #  prediction probabilities

# Plot non-normalized confusion matrix
plt.figure(1);
plot_confusion_matrix(cm_logreg_test, classes=target_names,
    title='Confusion matrixa teszt adatkészlethez (LogReg)');
plt.show();

plt.figure(2);
plot_confusion_matrix(cm_naive_bayes_test, classes=target_names,
   title='Confusion matrix teszt adatkészlethez (naiv Bayes)');
plt.show();

y_test_binarized = pre.label_binarize(y_test, target_names)

fpr_logreg, tpr_logreg, _ = metrics.roc_curve(y_test_binarized[:,6], yprobab_logreg[:,4]);
roc_auc_logreg = metrics.auc(fpr_logreg, tpr_logreg);

fpr_naive_bayes, tpr_naive_bayes, _ = metrics.roc_curve(y_test_binarized[:,6], yprobab_naive_bayes[:,4]);
roc_auc_naive_bayes = metrics.auc(fpr_naive_bayes, tpr_naive_bayes);

plt.figure(3);
lw = 1;
plt.plot(fpr_logreg, tpr_logreg, color='red',
         lw=lw, label='Logisztikus regresszió (area = %0.2f)' % roc_auc_logreg);
plt.plot(fpr_naive_bayes, tpr_naive_bayes, color='blue',
         lw=lw, label='Naive Bayes (area = %0.2f)' % roc_auc_naive_bayes);
plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--');
plt.xlim([0.0, 1.0]);
plt.ylim([0.0, 1.05]);
plt.xlabel('False Positive Mérték');
plt.ylabel('True Positive Mérték');
plt.title('A vevő működési jelleggörbéje');
plt.legend(loc="jobb alsó");
plt.show();