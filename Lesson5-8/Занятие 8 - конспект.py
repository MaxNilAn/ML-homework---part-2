import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
# from datetime import datetime, date, time


# ### Трехмерные точки и линии

fig = plt.figure()
ax = plt.axes(projection='3d')

# z1 = np.linspace(0, 15, 1000)
# y1 = np.cos(z1)
# x1 = np.sin(z1)

# ax.plot3D(x1, y1, z1, 'green')

# z2 = 15 * np.random.random(100)
# y2 = np.cos(z2) + 0.1 + np.random.random(100)
# x2 = np.sin(z2) + 0.1 + np.random.random(100)

# ax.scatter3D(x2, y2, z2, c=z2, cmap='Greens')

def f(x, y):
    return np.sin(np.sqrt(x**2 + y**2))
    
# x = np.linspace(-6, 6, 30)
# y = np.linspace(-6, 6, 30)
# X, Y = np.meshgrid(x, y)
# Z = f(X, Y)

# ax.contour3D(X, Y, Z, 40)
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')

# ax.view_init(60, 45)

# ax.scatter3D(X, Y, Z, c=Z, cmap='Greens')

# Каркасный график
# ax.plot_wireframe(X, Y, Z)

# Поверхностный график
# ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
# ax.set_title('Example')

r = np.linspace(0, 6, 20)
theta = np.linspace(-0.9 * np.pi, 0.8 * np.pi, 40)

R, Theta = np.meshgrid(r, theta)

X = R * np.sin(Theta)
Y = R * np.cos(Theta)

Z = f(X, Y)

ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')


# ### Триангуляция

ax = plt.axes(projection='3d')
theta = 2 * np.pi + np.random.random(1000)
r = 6 * np.random.random(1000)

x = r * np.sin(theta)
y = r * np.cos(theta)
z = f(x, y)

ax.scatter3D(x, y, z, c=z, cmap='viridis')

ax.plot_trisurf(x, y, z, cmap='viridis')

ax.view_init(30, 45)


# ### Seaborn
# - DataFrame (Matplotlib с Pandas)
# - более высокоуровневый

data = np.random.multivariate_normal([0, 0], [[5, 2], [2, 2]], size=2000)
data = pd.DataFrame(data, columns=['x', 'y'])

# print(data.head())

plt.figure()
plt.hist(data['x'], alpha=0.5)
plt.hist(data['y'], alpha=0.5)

plt.figure()
sns.kdeplot(data=data, fill=True)

iris = sns.load_dataset('iris')
# print(iris.head())

sns.pairplot(iris, hue='species', height=2.5)


# **Построение гистограмм**

tips = sns.load_dataset('tips')
# print(tips.head())

grid = sns.FacetGrid(tips, row='sex', col='day', hue='time')
grid.map(plt.hist, 'total_bill', bins=np.linspace(0,40,15))


# **Связь факторов**

sns.catplot(data=tips, x='day', y='total_bill', kind='box')


# **Совместное распределение для различных наборов данных**

sns.jointplot(data=tips, x='tip', y='total_bill', kind='hex')


# **Графики временных рядов**

planets = sns.load_dataset('planets')
# print(planets.head())

sns.catplot(data=planets, x='year', kind='count', hue='method', order=range(2005, 2015))


# **Изучение датасетов**
# 
# Сравнение числовых данных
# - Числовые пары

sns.pairplot(tips)


# - Тепловая карта
# 
# 0 - независимы, 1 - положительная зависимость, -1 - отрицательная зависимость

tips_corr = tips[['total_bill', 'tip', 'size']]
sns.heatmap(tips_corr.corr(), cmap='RdBu_r', annot=True, vmin=-1, vmax=1)


# - Диаграмма рассеяния

sns.scatterplot(data=tips, x='total_bill', y='tip', hue='sex')


# - Линейная регрессия

sns.regplot(data=tips, x='total_bill', y='tip')

# sns.relplot(data=tips, x='total_bill', y='tip', hue='sex')


# - Линейный график

sns.lineplot(data=tips, x='total_bill', y='tip')


# - Сводная диаграмма

sns.jointplot(data=tips, x='total_bill', y='tip')


# **Сравнение числовых и категориальных данных**
# 
# - Гистограмма

sns.barplot(data=tips, y='total_bill', x='day', hue='sex')


# - Точечная диаграмма

sns.pointplot(data=tips, y='total_bill', x='day', hue='sex')


# - Ящик "с усами"
# 
# Средняя линия - медиана (делит количество элементов пополам), крайние стороны ящика - квартили (отделяют 25% элементов на каждом промежутке)
#, части отрезков отделяют основное количество элементов, отдельные точки лежат за промежутоком от -3 сигма до +3 сигма

sns.boxplot(data=tips, y='total_bill', x='day')


# - Скрипичная диаграмма

data = np.random.multivariate_normal([0, 0], [[5, 2], [2, 2]], size=2000)
data = pd.DataFrame(data, columns=['x', 'y'])

# print(data.head())

plt.figure()
sns.kdeplot(data=data, fill=True)

plt.figure()

sns.violinplot(data=tips, y='total_bill', x='day')


# - Одномерная диаграмма рассеяния

sns.stripplot(data=tips, y='total_bill', x='day')

