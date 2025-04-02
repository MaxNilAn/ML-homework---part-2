#!/usr/bin/env python
# coding: utf-8

# ## Линейная регрессия
# ### Задача 
# На основе наблюдаемых точек построить прямую (некоторую гиперплоскость), которая отображает
# связь между двумя и более переменными.
# 
# Регрессия пытается "подогнать" функцию к наблюдаемым данным, чтобы спрогнозировать новые данные.
# 
# Линейная регрессия  - подгоняем данные к прямой линии, пытаемся установить линейную связь между переменными
# и предсказать новые данные.

# In[5]:


import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from numpy.linalg import inv, qr

features, target = make_regression(
    n_samples=100, n_features=1, n_informative=1, n_targets=1, noise=15, random_state=1
)

# print(features[:5],features.shape)
# print(target[:5],target.shape)

model = LinearRegression().fit(features, target)
plt.scatter(features,target)

x = np.linspace(features.min(), features.max(), 100)

plt.plot(x, model.coef_[0]*x + model.intercept_, color='red')


# ### Простая линейная регрессия
# Линейная -> линейная зависимость
# 
# **Плюсы:**
# - прогнозирование новых данных
# - анализ взаимного влияния переменных друг на друга
# 
# **Минусы:**
# - точки данных для обучения не будут лежать на одной прямой (шум) => область погрешности
# - не позволяет делать прогнозы вне диапазона имеющихся данных
# 
# Данные, на основание которых разрабатывается модель - это выборка из некоторой совокупности;
# хотелось бы, чтобы это была репрезентативная выборка.
# 
# #### Основные понятия:
# 
# **Остатки/отклонения/ошибки** - расстояния между точками данных и ближайшими по вертикали точками на прямой.
# 
# **Задача** состоит в минимизации остатков => разрыв между прямой и точками будет минимальным в некотором смысле:
# - сумма модулей остатков (функция недифференцируема в точке 0);
# - сумма квадратов остатков.
# 
# **Обучение модели** - минимизация функции потерь.
# 
# **Решения**:
# - численные - проще, доступнее (вычислительно), но решение неточное;
# - аналитические - точнее (мат. теория), но не всегда возможно.
# 
# #### 1) Аналитическое решение

# In[6]:


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
w_1 = ((n * sum(x[i]*y[i] for i in range(n))
       - sum(x)*sum(y))/
       (n*sum(x**2) - sum(x)**2))
w_0 = (sum(y[i] for i in range(n)))/n - w_1*(sum(x[i] for i in range(n)))/n
print(w_1,w_0)
# 2.4 0.8


# #### 2) Метод обратных матриц
# 
# w = (X^T*X)^(-1) * X^T * y

# In[9]:


x_1 = np.vstack([x, np.ones(len(x))]).T
w = inv(x_1.transpose() @ x_1) @ (x_1.transpose() @ y)
print(w)


# #### 3) Разложение матриц
# X = Q*R -> w = R^(-1) * Q^T * y
# 
# QR - разложение
# 
# Приближенные вычисления - позволяет минимизировать ошибку

# In[11]:


Q,R = qr(x_1)
w = inv(R).dot(Q.transpose()).dot(y)
print(w)


# #### 4) Градиентный спуск
# Метод оптимизации, где используются производные и итерации.
# 
# Частная производная (по одному из параметров) позволяет определить угловой коэффициент и изменение параметра
# выполянется в ту сторону, где он минимален /максимален.
# 
# Для больших угловых коэффициентов делается более широкий шаг. Ширина шага обычно вычисляется как доля
# от углового коэффициента -> скоростью обучения. Чем выше скорость, тем быстрее будет работать система,
# но за счет снижения точности. Чем ниже скорость, тем больше времени займет обучение, но точность будет выше.

# In[13]:


def f(x):
    return (x-3)**2 + 4
def df(x):
    return 2*x - 6
x = np.linspace(-10,10,100)
ax = plt.gca()
ax.xaxis.set_major_locator(plt.MultipleLocator(1))
# plt.plot(x,f(x))
plt.plot(x,df(x))
plt.grid()


# In[15]:


L = 0.001
iterations = 100_000

x = np.random.randint(0,5)

for i in range(iterations):
    d_x = df(x)
    x -= L*d_x
print(x,f(x))


# In[21]:


x = data[:,0]
y = data[:,1]

n = len(x)

w1 = 0.0
w0 = 0.0

L = 1e-3
iterations = 10000

for i in range(iterations):
    D_w0 = 2 * sum((-y[i] + w0 + w1*x[i]) for i in range(n))
    D_w1 = 2 * sum((x[i]*(-y[i] + w0 + w1*x[i])) for i in range(n))
    w0 -= L*D_w0
    w1 -= L*D_w1
print(w1,w0)


# In[23]:


w1 = np.linspace(-10,10,100)
w0 = np.linspace(-10,10,100)

def E(w1,w0,x,y):
    return sum((y[i] - (w0 + w1*x[i]))**2 for i in range(n))

W1, W0 = np.meshgrid(w1,w0)

EW = E(W1,W0,x,y)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(W1,W0,EW)

w1_fit = 2.4
w0_fit = 0.8

E_fit = E(w1_fit,w0_fit,x,y)
ax.scatter(w1_fit,w0_fit,E_fit,color='red')

