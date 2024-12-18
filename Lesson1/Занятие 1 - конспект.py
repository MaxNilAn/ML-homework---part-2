# import numpy as np
# import sys
# import array

# Типы данных Python

# x = 1
# print(type(x))
# print(sys.getsizeof(x))

# Динамическая типизация
# x = 'hello'
# print(type(x))

# x = True
# print(type(x))

# Списки (массивы)
# l1 = list([])
# print(sys.getsizeof(l1))

# l2 = list([1, 2, 3])
# print(sys.getsizeof(l2))

# l3 = list([1, '2', True])
# print(l3, sys.getsizeof(l2))

# Массив однотипных элементов
# a1 = array.array('i', [1, 2, 3])
# print(type(a1), sys.getsizeof(a1))

# Массив numpy
# a = np.array([1, 2, 3, 4, 5])
# print(a, type(a))

# Повышающее приведение типов
# a = np.array([1.23, 2, 3, 4 ,5])
# print(a, type(a))

# Определение типа для элементов массива
# a = np.array([1.23, 2, 3, 4 ,5], dtype=int)
# print(a, type(a))

# a = np.array([range(i, i+3) for i in [2,4,6]])
# print(a, type(a))

# Массив нулей
# a = np.zeros(10, dtype=int)
# print(a, type(a))

# Массив единиц
# print(np.ones((3, 5), dtype=float))

# Массив из одинаковых элементов
# print(np.full((4, 5), 3.1415))

# Массив из последовательных чисел с шагом
# print(np.arange(0, 20, 2))

# Единичная матрица
# print(np.eye(4))

### МАССИВЫ

# np.random.seed(1)

# Случайные числа
# x1 = np.random.randint(10, size=3)
# x2 = np.random.randint(10, size=(3,2))
# x3 = np.random.randint(10, size=(3,2,1))

# print(x1)
# print(x2)
# print(x3)

# Размерности и размеры
# print(x1.ndim, x1.shape, x1.size)
# print(x2.ndim, x2.shape, x2.size)
# print(x3.ndim, x3.shape, x3.size)

## Индекс (с 0)

# a = np.array([1, 2, 3, 4, 5])
# print(a[0], a[-2])
# a[1] = 20
# print(a)

# a = np.array([[1, 2], [3, 4]])
# print(a)
# print(a[0, 0], a[-1, -1])
# a[1, 0] = 100
# print(a)

# a = np.array([1, 2, 3, 4])
# b = np.array([1.0, 2, 3, 4])
# print(a, b)
# a[0] = 10
# print(a)
# a[0] = 10.123
# print(a)

## Срез [start:end:step] [0:shape:1]
# a = np.array([1, 2, 3, 4, 5, 6])
# print(a[:3], a[3:], a[1:5], a[1:-1], a[1::2])
# print(a[::-1])

# b = a[:3]
# print(b)
# b[0] = 100
# print(a)

# a = np.arange(1, 13)
# print(a)
# print(a.reshape(2, 6))
# print(a.reshape(3, 4))

# Объединение массивов
# x = np.array([1, 2, 3])
# y = np.array([4, 5])
# z = np.array([6])
# print(np.concatenate([x, y, z]))

# x = np.array([1, 2, 3])
# y = np.array([4, 5, 6])
# r1 = np.vstack([x,y])
# print(r1)

# print(np.hstack([r1, r1]))
# print(np.dstack([r1, r1]))

### Вычисления с массивами

## Векторизированная операция - независимо к каждому элементу

# x = np.arange(10)
# print(x)
# print(x*2 + 1)

# Универсальные функции

# print(np.add(np.multiply(x, 2),1))

# -; -; /; //; **; %
# np.abs, sin/cos/tan, exp, log

# Вывод в другую переменную
# x = np.arange(5)
# y = np.zeros(10)
# print(np.multiply(x, 10, out=y[::2]))
# print(y)

# Получение массива из промежуточных результатов
# x = np.arange(1,5)
# print(x)
# print(np.add.accumulate(x))

# Таблица из попарных операций
# x = np.arange(1, 10)
# print(np.multiply.outer(x, x))