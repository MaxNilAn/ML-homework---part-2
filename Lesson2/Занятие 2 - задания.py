import numpy as np

## 1. Что надо изменить в примере, чтобы он заработал? (транслирование)
print('Задание 1')
a = np.ones((3,2))
b = np.arange(3)
print('Исходные массивы:\n','a = ',a,'\n','b = ',b)
print('Вариант 1 - изменить форму одного из массивов:')
a1 = a.reshape(2,3)
print('a* = ',a1)
print('a* + b = ',a1 + b)
print('Вариант 2 - разбить первый массив по одному из измерений, выполнить транслирование для каждого сечения, объединить полученные массивы:')
a1, a2 = np.hsplit(a,2)
print('a1 = ',a1,'\na2 = ',a2)
print('a + b = (a1 + b),(a2 + b)')
print('a + b = ',np.vstack([a1+b,a2+b]))

## 2. Пример для y. Вычислить количество элементов (по обоим размерностям), значения которых больше 3 и меньше 9
print('\nЗадание 2')
y = np.array([[1,2,3,4,5],[6,7,8,9,10]])
print(y)
print('Количество элементов > 3 и < 9:', np.sum((y > 3) & (y < 9)))
print('Количество элементов > 3 и < 9 (столбцы):', np.sum((y > 3) & (y < 9), 0))
print('Количество элементов > 3 и < 9 (строки):', np.sum((y > 3) & (y < 9), 1))