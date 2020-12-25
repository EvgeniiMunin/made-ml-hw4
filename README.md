# made-ml-hw4

# Домашнее задание по курсу Машинное обучение Академии больших данных MADE

Команда разработчиков:
|                |Группа |Гитхаб                          |Роль в проекте |
|----------------|-------|--------------------------------|---------------|
|Авилов Илья     |DS-11  |https://github.com/Ilya2567     |DS, Frontend   |
|Дякин Николай   |ML-11  |https://github.com/nickdndev    |DS, Frontend   |
|Мунин Евгений   |ML-12  |https://github.com/EvgeniiMunin |DS, ML, Frontend   |
|Орхан Гаджилы   |DS-12  |https://github.com/Fianketto    |DS, PM         |
|Стариков Андрей |ML-12  |https://github.com/andyst75     |DS, ML, DevOps |

[Демоверсия проекта](https://made-ml-hw4.herokuapp.com/)

## Тема проекта: Предсказание валютных котировок


## Математическая постановка задачи

|Model           |MAE val                   |
|----------------|-------------------------------|
|Naive model: Moving average (40 days)|1355|
|Linear Regression + Lag features (40 days)| 704|
|XGBRegressor| 2173|
|LSTM + Sliding window (40 days)| 1949|

## Заключения по тестированию моделей
- Точность модели скользящего среднего является хорошим ориентиром для валидации более сложных моделей, который сложно перебить.
- Наилучший результат показа модель линейной регрессии с применением лаговых признаков, построенных из целевой переменной курса закрытия пары валют.
- Усложнение архитектуры модели не приводит к улучшению по сравнению с линейной моделью.
