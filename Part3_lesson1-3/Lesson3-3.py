#!/usr/bin/env python
# coding: utf-8

# ## Переобучение и дисперсия
# Нахождение оптимальной прямой при минимизации квадратов остатков не гарантирует удачность
# линейной регрессии.
# 
# **Цель** - не минимизация, а возможность делать правильные предсказания на новых данных.
# 
# Кривые, проходящие точно через точки данных для обучения, являются **переобученными**.
# Переобученные модели очень чувствительны к выбросам - точкам, сильно отклоняющимся от остальных данных;
# в прогнозах будет очень высокая дисперсия, поэтому к моделям специально добавляются смещения.
# 
# **Смещение** модели означает, что предпочтение отдается определенной схеме (например, прямая линия),
# а не графику со сложной структурой, который минимизирует сумму квадратов остатков.
# Если в модель добавить смещение, то есть риск **недообучения**.
# Возникает необходимость в балансировке между минимизацией функцией потерь и смещением для предотвращения переобучения.
# 
# #### Варианты смещенной линейной регрессии:
# - гребневая регрессия (ridge): добавляет смещение в виде штрафа, из-за чего хуже идет подгонка
# - лассо-регрессия: нерелевантные признаки удаляются, что уменьшает размерность
# 
# Механически применить регрессию к данным, сделать на основе полученной модели прогноз и
# думать, что все в порядке, - нельзя.
# 
# 
# #### Виды градиентного спуска:
# 
# **Пакетный** градиентный спуск: для работы используются все доступные обучающие данные.
# 
# **Стохастический** градиентный спуск: на каждой итерации обучаемся только на одной выборке из данных.
# - сокращение числа вычислений
# - вносим смещение => боремся с переобучением
# 
# **Минипакетный** градиентный спуск: на каждой итерации используется несколько выборок.

# In[3]:


import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import matplotlib.pyplot as plt

data = np.array([
        [1, 5],
        [2, 7],
        [3, 7],
        [4, 10],
        [5, 11],
        [6, 14],
        [7, 17],
        [8, 19],
        [9, 22],
        [10, 28],
])

x = data[:,0]
y = data[:,1]

n = len(x)

w1 = 0.0
w0 = 0.0

L = 1e-3
# размер выборки
sample_size = 1

iterations = 10000
for i in range(iterations):
    idx = np.random.choice(n, sample_size, replace=False)
    D_w0 = 2 * sum((-y[i] + w0 + w1*x[i]) for i in idx)
    D_w1 = 2 * sum((x[i]*(-y[i] + w0 + w1*x[i])) for i in idx)
    w0 -= L*D_w0
    w1 -= L*D_w1
print(w1,w0)


# Как оценить, насколько сильно промахиваются прогнозы при использовании линейной регрессии?
# 
# Для оценки степени взаимосвязи между двумя переменным
# и использовался линейный **коэффициент корреляции**.
# 
# Он позволяет понять, есть ли связь между двумя переменными.

# In[5]:


data_df = pd.DataFrame(data)
print(data_df.corr(method="pearson"))

# data_df[1] = data_df[1].values[::-1]
# print(data_df.corr(method="pearson"))


# ### Обучающие и тестовые выборки
# Основной метод борьбы с переобучением, заключается в том, что набор данных делится на
# обучающую и тестовую выборки.
# 
# Во всех видах машинного обучения с учителем это встречается. 
# 
# Обычная пропорция: 2/3 - обучение, 1/3 - тест. Другие варианты: 4/5, 1/5; 9/10, 1/10.

# In[7]:


X = data_df.values[:,0,np.newaxis]
Y = data_df.values[:,1,np.newaxis]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3)
# print(X_train, X_test, Y_train, Y_test)

model = LinearRegression()
model.fit(X_train, Y_train)

# Коэффициент детерминации
r = model.score(X_test, Y_test)
print(r)


# **Коэффициент детерминации r** - в отсутствии разделения на выборки квадрат корреляции.
# При разделении равен: r^2 = 1 - сумма(y_i - предсказанное значение)^2/
# сумма(y_i - среднее значение по всем i)^2
# 
# 0 < r < 1. Чем ближе к 1, тем лучше. 
# 
# При выполнении различных разбиений и обучении - перекрестная проверка.

# In[11]:


X = data_df.values[:,0,np.newaxis]
Y = data_df.values[:,1,np.newaxis]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3)
# print(X_train, X_test, Y_train, Y_test)

# 3-х кратная перекрестная валидация
kfold = KFold(n_splits=3, random_state=1, shuffle=True)

model = LinearRegression()
# model.fit(X_train, Y_train)

# Среднеквадратические ошибки
results = cross_val_score(model, X, Y, cv=kfold)

print(results)
print(results.mean(), results.std())
# Коэффициент детерминации
# r = model.score(X_test, Y_test)


# Метрики показывают, насколько ЕДИНООБРАЗНО ведет себя модель на разных выборках.
# Можно выполнить k-кратную или даже поэлементную валидацию.
# 
# Случайная валидация с помощью перемешивания позволяет многократно использовать одни и те же данные.

# In[48]:


data_df = pd.read_csv('./multiple_independent_variable_linear.csv')
# print(data_df.head())

X = data_df.values[:,:-1]
Y = data_df.values[:,-1]

model = LinearRegression().fit(X,Y)

# print(model.coef_,model.intercept_)

x1 = X[:,0]
x2 = X[:,1]
y = Y

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(x1, x2, y)

x1_ = np.linspace(min(x1), max(x1), 100)
x2_ = np.linspace(min(x2), max(x2), 100)

X1_, X2_ = np.meshgrid(x1_, x2_)

Y_ = model.intercept_ + model.coef_[0]*X1_ + model.coef_[1]*X2_

ax.plot_surface(X1_, X2_, Y_, cmap="Greys", alpha=0.5)

