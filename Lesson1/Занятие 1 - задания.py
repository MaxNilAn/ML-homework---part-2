import array
import sys
import numpy as np

# Задание 1. Коды типов для array: 'u' - unicode character, 'f' - float, 'd' - double + разные варианты для integer ('h', 'i', 'l', 'q')

# Задание 2. Array с элементами другого типа
print('Задание 2')
a1 = array.array('u', ['a', 'b'])
print(a1, type(a1), sys.getsizeof(a1))

# Задание 3. Массив с 5 значениями, располагающимися через равные интервалы в диапазоне от 0 до 1
print('\nЗадание 3')
x1 = np.arange(0, 1.01, 0.25)
print(x1)

# Задание 4. Массив с 5 равномерно распределенными случайными значениями в диапазоне от 0 до 1
print('\nЗадание 4')
x1 = np.random.uniform(0, 1, 5)
print(x1)

# Задание 5. Массив с 5 нормально распределенными случайными значениями с мат. ожиданием 0 и дисперсией 1
print('\nЗадание 5')
x1 = np.random.uniform(0, 1, 5)
print(x1)

# Задание 6. Массив с 5 случайными целыми числами в [0, 10)
print('\nЗадание 6')
x1 = np.random.randint(0, 10, 5)
print(x1)

# Задание 7. Код для создания срезов массива 3 на 4
print('\nЗадание 7\nИсходный массив:')
x1 = np.arange(1,13).reshape(3,4)
print(x1)
print('Первые две строки и три столбца:\n', x1[:2,:3])
print('Первые три строки и второй столбец:\n', x1[:3,1])
print('Все строки и столбцы в обратном порядке:\n', x1[::-1,::-1])
print('Второй столбец:\n', x1[:,1])
print('Третья строка:\n', x1[2, :])

# Задание 8. Срез-копия
print('\nЗадание 8')
a = np.arange(5)
b = np.array(a[1:4])
print('Исходные массивы: a = ', a, ', b = ', b)
b[1] = 10
print('Массивы после изменения b: a = ', a, ', b = ', b)

# Задание 9. Использование newaxis (добавление размерности)
print('\nЗадание 9')
x = np.arange(6)
print('Исходный массив:\n', x)
row = x[np.newaxis, :]
print('Вектор-строка:\n', row)
col = x[:, np.newaxis]
print('Вектор-столбец:\n', col)

# Задание 10. Использование dstack (объединение массивов в глубину)
print('\nЗадание 10')
x = np.arange(6).reshape(2,3)
print('x:\n', x)
print('\ndstack([x, x]):\n', np.dstack([x,x]))

# Задание 11. Использование split, vsplit, hsplit, dsplit (разбиение массива по разным направлениям)
print('\nЗадание 11')
x = np.arange(8).reshape(2, 2, 2)
print('x:\n', x)
print('\nsplit(x, 2):\n', np.split(x, 2))
print('\nhsplit(x, 2):\n', np.hsplit(x, 2))
print('\nvsplit(x, 2):\n', np.vsplit(x, 2))
print('\ndsplit(x, 2):\n', np.dsplit(x, 2))

# Задание 12. Использование универсальных функций -; -; /; //; **; %
print('\nЗадание 12')
x = np.arange(10)
print(' x: ', x)
print(' -x: ', -x)
print(' x-2: ', x-2)
print(' x/3: ', x/3)
print(' x//3: ', x//3)
print(' x**2: ', x**2)
print(' x%3: ', x%3)