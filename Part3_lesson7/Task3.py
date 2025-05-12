#!/usr/bin/env python
# coding: utf-8

# ## Метод k средних

# In[51]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans

iris = sns.load_dataset('iris')

data = iris[['sepal_length','sepal_width','petal_width','species']]
data_df = data[(data['species'] == 'virginica') | (data['species'] == 'versicolor')]

X = data_df[['sepal_length','sepal_width','petal_width']]
y = data_df['species']

data_df_virginica = data_df[data_df['species'] == 'virginica']
data_df_versicolor = data_df[data_df['species'] == 'versicolor']


# In[53]:


kmeans = KMeans(n_clusters=2)
kmeans.fit(X)
pred = kmeans.predict(X);


# In[57]:


fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(data_df_virginica['sepal_length'],data_df_virginica['sepal_width'],data_df_virginica['petal_width'],c='red',label='virginica',s=30)
ax.scatter(data_df_versicolor['sepal_length'],data_df_versicolor['sepal_width'],data_df_versicolor['petal_width'],c='purple',label='versicolor',s=30)
ax.set_xlabel('sepal_length')
ax.set_ylabel('sepal_width')
ax.set_zlabel('petal_width')
plt.title('Original data')
ax.legend()
plt.savefig('Task3_image1.png')

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel('sepal_length')
ax.set_ylabel('sepal_width')
ax.set_zlabel('petal_width')
ax.scatter(data_df['sepal_length'],data_df['sepal_width'],data_df['petal_width'], c=pred, s=30, cmap='rainbow')
centers = kmeans.cluster_centers_
plt.title('Prediction')
ax.scatter(centers[:, 0], centers[:, 1],  centers[:, 2], c='black', s=200, label='centers',alpha=0.5)
ax.legend()
plt.savefig('Task3_image2.png')
plt.show()

