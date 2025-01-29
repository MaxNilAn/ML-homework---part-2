import numpy as np
import pandas as pd

# 1. Разобраться как использовать мультииндексные ключи в данном примере
print('Задание 1:\n')
index = [
    ('city_1',2010),
    ('city_1',2020),
    ('city_2',2010),
    ('city_2',2020),
    ('city_3',2010),
    ('city_3',2020)
]

population = [
    101,
    1010,
    201,
    2010,
    102,
    1020,
]

pop = pd.Series(population, index=index)
index1 = pd.MultiIndex.from_tuples(index)

pop_df = pd.DataFrame(
    {
        'total':pop,
        'something':[
                11,
                12,
                13,
                14,
                15,
                16
        ]
    },
    index=index1
)
print('Исходный DataFrame:\n', pop_df, '\n')
pop_df_1 = pop_df.loc[['city_1'],'something']
pop_df_2 = pop_df.loc[['city_1', 'city_3'],['total', 'something']]
pop_df_3 = pop_df.loc[['city_1', 'city_3'],'something']
print('Выборка 1:\n', pop_df_1, '\n')
print('Выборка 2:\n', pop_df_2, '\n')
print('Выборка 3:\n', pop_df_3, '\n')

# 2. Из получившихся данных выбрать данные по 
# - 2020 году (для всех столбцов)
# - job_1 (для всех строк)
# - для city_1 и job_2 
print('Задание 2:\n')

index = pd.MultiIndex.from_product(
    [
        ['city_1','city_2'],
        [2010,2020]
    ],
    names=['city','year']
)

columns = pd.MultiIndex.from_product(
    [
        ['person_1','person_2','person_3'],
        ['job_1','job_2']
    ],
    names=['worker','job']
)

rng = np.random.default_rng(1)
data = rng.random((4, 6))

data_df = pd.DataFrame(data, index=index, columns=columns)
print('Исходный DataFrame:\n', data_df, '\n')

print('Данные по 2020 году для всех столбцов:\n', data_df.loc[(slice(None), 2020), :], '\n')
print('Данные с job_1 для всех строк:\n', data_df.loc[:, (slice(None),'job_1')], '\n')
print('Данные с city_1 и job_2:\n', data_df.loc[('city_1', slice(None)), (slice(None),'job_2')], '\n')

# 3. Взять за основу DataFrame со следующей структурой (из задания 2)
# Выполнить запрос на получение следующих данных
# - все данные по person_1 и person_3
# - все данные по первому городу и первым двум person-ам (с использование срезов)
#
# Приведите пример (самостоятельно) с использованием pd.IndexSlice
print('Задание 3:\n')
# pd.IndexSlice позволяет избавиться от конструкций slice(None)
idx = pd.IndexSlice
print('Данные по person_1 и person_3:\n', data_df.loc[:, idx[['person_1','person_3'],:]], '\n')
print('Данные по первому городу и первым двум person-ам:\n', data_df.loc[idx['city_1',:], idx['person_1':'person_2',:]], '\n')

# 4. Привести пример использования inner и outer джойнов (для возможности использования join объект Series заменен на DataFrame)
print('Задание 4:\n')
# Создание объектов DataFrame, которые имеют столбцы с общими названиями
df1 = pd.DataFrame([['a', 1, 11], ['b', 2, 12], ['c', 3, 13]], columns=['one', 'two','three'])
df2 = pd.DataFrame([['d', 4, 14], ['e', 5, 15], ['f', 6, 16]], columns=['one', 'four', 'three'])
print('DataFrame1:\n', df1, '\n')
print('DataFrame2:\n', df2, '\n')
# Внешнее объединение DataFrame - объединение стобцов
print('Объединение:\n', pd.concat([df1, df2], join='outer'),'\n')
# Внутреннее объединение DataFrame - пересечение стобцов
print('Пересечение:\n', pd.concat([df1, df2], join='inner'))