#!/usr/bin/env python
# coding: utf-8

# ## Метод опорных векторов

# In[141]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.svm import SVC

iris = sns.load_dataset('iris')

data = iris[['sepal_length','sepal_width','petal_width','species']]
data_df = data[(data['species'] == 'virginica') | (data['species'] == 'versicolor')]

X = data_df[['sepal_length','sepal_width','petal_width']]
y = data_df['species']

data_df_virginica = data_df[data_df['species'] == 'virginica']
data_df_versicolor = data_df[data_df['species'] == 'versicolor']


# In[143]:


model = SVC(kernel='linear', C=10000)
model.fit(X, y)


# In[157]:


fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(data_df_virginica['sepal_length'],data_df_virginica['sepal_width'],data_df_virginica['petal_width'],c='red',label='virginica')
ax.scatter(data_df_versicolor['sepal_length'],data_df_versicolor['sepal_width'],data_df_versicolor['petal_width'],c='green',label='versicolor')

x_p = np.linspace(min(data_df['sepal_length']), max(data_df['sepal_length']), 100)
y_p = np.linspace(min(data_df['sepal_width']), max(data_df['sepal_width']), 100)
z_p = np.linspace(min(data_df['petal_width']), max(data_df['petal_width']), 100)

X, Y = np.meshgrid(x_p,y_p)

w = model.coef_[0]
b = model.intercept_[0]
Z = (-w[0] * X - w[1] * Y - b) / w[2]
ax.plot_surface(X, Y, Z, alpha=0.7)
ax.set_xlabel('sepal_length')
ax.set_ylabel('sepal_width')
ax.set_zlabel('petal_width')
ax.legend()
ax.view_init(15, 35)
plt.savefig('Task1_image.png')
plt.show()

