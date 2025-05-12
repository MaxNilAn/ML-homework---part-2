#!/usr/bin/env python
# coding: utf-8

# ## Метод главных компонент

# In[112]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

iris = sns.load_dataset('iris')

data = iris[['sepal_length','sepal_width','petal_width','species']]
data_virginica = data[data['species'] == 'virginica'].drop(columns='species')
data_versicolor = data[data['species'] == 'versicolor'].drop(columns='species')


# In[116]:


x = data_virginica['sepal_length']
y = data_virginica['sepal_width']
z = data_virginica['petal_width']

p = PCA(n_components=3)
p.fit(data_virginica)
X_p = p.transform(data_virginica)

p1 = PCA(n_components=1)
p1.fit(data_virginica)
X_p1 = p1.transform(data_virginica)

X_p1_new = p1.inverse_transform(X_p1)


# In[118]:


fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(data_virginica['sepal_length'],data_virginica['sepal_width'],data_virginica['petal_width'],c='red')
ax.plot(
    [p.mean_[0], p.mean_[0] + p.components_[0][0] * np.sqrt(p.explained_variance_[0])*3],
    [p.mean_[1], p.mean_[1] + p.components_[0][1] * np.sqrt(p.explained_variance_[0])*3],
    [p.mean_[2], p.mean_[2] + p.components_[0][2] * np.sqrt(p.explained_variance_[0])*3],
    
)
ax.plot(
    [p.mean_[0], p.mean_[0] + p.components_[1][0] * np.sqrt(p.explained_variance_[1])*3],
    [p.mean_[1], p.mean_[1] + p.components_[1][1] * np.sqrt(p.explained_variance_[1])*3],
    [p.mean_[2], p.mean_[2] + p.components_[1][2] * np.sqrt(p.explained_variance_[1])*3]
)
ax.plot(
    [p.mean_[0], p.mean_[0] + p.components_[2][0] * np.sqrt(p.explained_variance_[2])*3],
    [p.mean_[1], p.mean_[1] + p.components_[2][1] * np.sqrt(p.explained_variance_[2])*3],
    [p.mean_[2], p.mean_[2] + p.components_[2][2] * np.sqrt(p.explained_variance_[2])*3]
)
ax.scatter(X_p1_new[:,0], X_p1_new[:,1], X_p1_new[:,2], c='black', alpha=0.4)
ax.set_xlabel('sepal_length')
ax.set_ylabel('sepal_width')
ax.set_zlabel('petal_width')
plt.title('virginica')
plt.savefig('Task2_image1.png')
plt.show()


# In[122]:


x = data_versicolor['sepal_length']
y = data_versicolor['sepal_width']
z = data_versicolor['petal_width']

p = PCA(n_components=3)
p.fit(data_versicolor)
X_p = p.transform(data_versicolor)

p1 = PCA(n_components=1)
p1.fit(data_versicolor)
X_p1 = p1.transform(data_versicolor)

X_p1_new = p1.inverse_transform(X_p1)


# In[124]:


fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(data_versicolor['sepal_length'],data_versicolor['sepal_width'],data_versicolor['petal_width'],c='red')
ax.plot(
    [p.mean_[0], p.mean_[0] + p.components_[0][0] * np.sqrt(p.explained_variance_[0])*3],
    [p.mean_[1], p.mean_[1] + p.components_[0][1] * np.sqrt(p.explained_variance_[0])*3],
    [p.mean_[2], p.mean_[2] + p.components_[0][2] * np.sqrt(p.explained_variance_[0])*3],
    
)
ax.plot(
    [p.mean_[0], p.mean_[0] + p.components_[1][0] * np.sqrt(p.explained_variance_[1])*3],
    [p.mean_[1], p.mean_[1] + p.components_[1][1] * np.sqrt(p.explained_variance_[1])*3],
    [p.mean_[2], p.mean_[2] + p.components_[1][2] * np.sqrt(p.explained_variance_[1])*3]
)
ax.plot(
    [p.mean_[0], p.mean_[0] + p.components_[2][0] * np.sqrt(p.explained_variance_[2])*3],
    [p.mean_[1], p.mean_[1] + p.components_[2][1] * np.sqrt(p.explained_variance_[2])*3],
    [p.mean_[2], p.mean_[2] + p.components_[2][2] * np.sqrt(p.explained_variance_[2])*3]
)
ax.scatter(X_p1_new[:,0], X_p1_new[:,1], X_p1_new[:,2], c='black', alpha=0.4)
ax.set_xlabel('sepal_length')
ax.set_ylabel('sepal_width')
ax.set_zlabel('petal_width')
plt.title('versicolor')
plt.savefig('Task2_image2.png')
plt.show()

