import numpy as np
import pandas as pd

# 1. Привести различные способы создания объектов типа Series
print('Задание 1:')
# - списки Python или массивы NumPy
s1 = pd.Series([1, 2, 3])
s11 = pd.Series(np.array([4, 5, 6]))
print(s1,'\n')
print(s11,'\n')
# - скалярные значения
s2 = pd.Series(1.5)
print(s2,'\n')
# - словари
s3 = pd.Series({'a': 3, 'b':7})
print(s3,'\n')

# 2. Привести различные способы создания объектов типа DataFrame
print('Задание 2:')
# - через объекты Series
df1 = pd.DataFrame([s1, s11])
print(df1,'\n')
# - списки словарей
df2 = pd.DataFrame([{'a':2.5, 'b':1.7}, {'a':-3.3, 'b':5}])
print(df2,'\n')
# - словари объектов Series
df3 = pd.DataFrame({'col1':s1, 'col2':s11})
print(df3,'\n')
# - двумерный массив NumPy
df4 = pd.DataFrame(np.array([[1, 2], [3, 4]]))
print(df4,'\n')
# - структурированный массив Numpy
df5 = pd.DataFrame(np.array([('first', 2), ('second', 7)], dtype=[('text', 'U10'), ('num', 'i4')]))
print(df5,'\n')

# 3. Объедините два объекта Series с неодинаковыми множествами ключей (индексов) так, чтобы вместо NaN было установлено значение 1
print('Задание 3:')
pop = pd.Series({
    'city_1': 1001,
    'city_2': 1002,
    'city_3': 1003,
    'city_41': 1004,
    'city_51': 1005
})

area = pd.Series({
    'city_1': 9991,
    'city_2': 9992,
    'city_3': 9993,
    'city_42': 9994,
    'city_52': 9995
})

data = pd.DataFrame({'area1': area,'pop1': pop}).fillna(1)
print(data,'\n')

# 4. Переписать пример с транслированием для DataFrame так, чтобы вычитание происходило по СТОЛБЦАМ
print('Задание 4:')
rng = np.random.default_rng()
A = rng.integers(0,10,(3,4))
df = pd.DataFrame(A, columns = ['a', 'b', 'c', 'd'])
print('Исходный DataFrame:')
print(df)

print('DataFrame, в котором из всех столбцов вычитается первый:')
print(df - df.iloc[:,0].values[:,np.newaxis],'\n')

# 5. На примере объектов DataFrame продемонстрируйте использование методов ffill() и bfill()
print('Задание 5:')
dff = pd.DataFrame({'a':[1, np.nan, np.nan, 2, 3], 'b':[np.nan, 4, np.nan, 5, np.nan]})
print('Исходный DataFrame:')
print(dff)
print('Использование ffill() - замена NaN предыдущим известным значением (вперед):')
print(dff.ffill())
print('Использование bfill() - замена NaN следующим известным значением (назад):')
print(dff.bfill())