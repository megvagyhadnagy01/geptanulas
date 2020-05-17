# -*- coding: utf-8 -*-
"""
Created on Mon May 18 00:40:06 2020

@author: megva
"""



import numpy as np;  # importing numerical computing package
from urllib.request import urlopen;  # importing url handling
from matplotlib import pyplot as plt;  # importing the MATLAB-like plotting tool
from sklearn import cluster;  # importing clustering algorithms
from sklearn import metrics;  # importing cluster metrics




# Reading the dataset

url = 'https://raw.githubusercontent.com/megvagyhadnagy01/geptanulas/master/ticdata2000.txt';
label_row_number = 1;
attribute_cols = 333;
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



link = 'ward';  
ward_cluster = cluster.AgglomerativeClustering(distance_threshold=0, 
                            n_clusters=None,linkage=link);
ward_cluster.fit(X);



fig = plt.figure(1);
plt.title('Az adatpontok szóródása címkével');
plt.xlabel('X');
plt.ylabel('Y');
plt.scatter(X[:,0],X[:,1],s=50,c=y);
plt.show();


K = 9;
single_cluster = cluster.AgglomerativeClustering(n_clusters=K,linkage=link);
single_cluster.fit(X);
ypred_single = single_cluster.labels_;
db_single = metrics.davies_bouldin_score(X,ypred_single);
cm_single = metrics.cluster.contingency_matrix(y,ypred_single);


fig = plt.figure(2);
plt.title('Az adatpontok szóródása egycsatornás kapcsolócsoportokka');
plt.xlabel('X');
plt.ylabel('Y');
plt.scatter(X[:,0],X[:,1],s=50,c=ypred_single);
plt.show();

Max_K = 90;
SSE = np.zeros((Max_K-2));
DB = np.zeros((Max_K-2));
for i in range(Max_K-2):
    n_c = i+2;
    kmeans = cluster.KMeans(n_clusters=n_c, random_state=2020);
    kmeans.fit(X);
    ypred = kmeans.labels_;
    SSE[i] = kmeans.inertia_;
    DB[i] = metrics.davies_bouldin_score(X,ypred);
    
fig = plt.figure(3);
plt.title('A hibagörbe négyzeteinek összege');
plt.xlabel('Klaszterek száma');
plt.ylabel('SSE');
plt.plot(np.arange(2,Max_K),SSE, color='red')
plt.show();

fig = plt.figure(4);
plt.title('Davies-Bouldin pontszámgörbe');
plt.xlabel('Klaszterek száma');
plt.ylabel('DB index');
plt.plot(np.arange(2,Max_K),DB, color='blue')
plt.show();
