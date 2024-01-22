# Исследование и разработка системы для автоматической идентификации предпочтений пользователей в электронной коммерции

<a target="_blank" href="https://colab.research.google.com/github/LisiyLexa/Lab3-Recommendation-System/blob/main/recommendation_system.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

## Введение
Хорошо разработанная система рекомендаций поможет компаниям улучшить восприятие своих покупателей на веб-сайте и приведет к лучшему привлечению и удержанию клиентов.

Система рекомендаций, которую я разработал ниже, основана на путешествии нового клиента с момента, когда он впервые заходит на веб-сайт компании, до момента, когда он совершает повторные покупки.

Система рекомендаций состоит из 3 частей, основанных на бизнес-контексте:

[Часть I](#Часть-I): Система, основанная на популярности продукта, ориентированная на новых клиентов

[Часть II](#Часть-II): Система коллаборативной фильтрации, основанная на истории покупок клиента и оценках, предоставленных другими пользователями, которые покупали похожие товары

[Часть III](#Часть-III): Случай, когда компания впервые создает свой веб-сайт электронной коммерции без какого-либо рейтинга продукта

Когда новый клиент без какой-либо истории предыдущих покупок впервые посещает веб-сайт электронной коммерции, ему рекомендуются самые популярные товары, продаваемые на веб-сайте компании. Как только пользователь совершает покупку, система рекомендаций обновляет и рекомендует другие продукты на основе истории покупок и оценок, предоставленных другими пользователями на веб-сайте.

## Используемые библиотеки
- numpy
- pandas
- sklearn
  - decomposition
  - feature_extraction.text
  - neighbors
  - cluster
## Наборы данных
#### Часть 1, 2: [Amazon product review dataset](https://www.kaggle.com/skillsmuggler/amazon-ratings)
#### Часть 3: [Home Depot's dataset with product dataset](https://www.kaggle.com/c/home-depot-product-search-relevance/data)

## Часть I
### Система, основанная на популярности продукта, ориентированная на новых клиентов
* Ориентация на популярность - отличная стратегия для привлечения новых клиентов с помощью самых популярных продуктов, продаваемых на веб-сайте компании, и очень полезна для холодного запуска системы рекомендаций.

Вид данных:

| UserId | ProductId | Rating | Timestamp |
| --- | --- | --- | --- |
|A39HTATAQ9V7YF	|0205616461	|5.0	|1369699200
|A3JM6GV9MNOF9X	|0558925278	|3.0	|1355443200
|A1Z513UWSAAO0F	|0558925278	|5.0	|1404691200
|A1WMRR494NWEWV	|0733001998	|4.0	|1382572800
|A3IAAVS479H7M7	|0737104473	|1.0	|1274227200

Всего записей: 2023070

Топ-10 продуктов по покупкам:

| ProductId | Buys |
|---|---|
|B001MA0QY2	|7533
|B0009V1YR8	|2869
|B0043OYFKU	|2477
|B0000YUXI0	|2143
|B003V265QW	|2088
|B000ZMBSPE	|2041
|B003BQ6QXK	|1918
|B004OHQR1Q	|1885
|B00121UVU0	|1838
|B000FS05VG	|1589

![image](https://github.com/LisiyLexa/Lab3-Recommendation-System/assets/81087786/5cfca226-411c-4c5b-b52a-0b1d2a54001f)

## Часть II

### Система коллаборативной фильтрации
* Рекомендуем товары клиентам, опираясь на их предыдущие покупки и схожесть оценок с другими покупателями, которые приобрели аналогичные товары, и сравниваем эти оценки с оценками этого конкретного клиента.
* Метод коллаборативной фильтрации здесь подходит, поскольку он помогает прогнозировать продукты для конкретного пользователя, выявляя закономерности на основе предпочтений из множества пользовательских данных.

_В данной работе в демонстрационных целях были использованны только 10000 случайных записей из таблицы_

Таблица полезности (первые 5 строк):

| ProductId\UserID |A01254332UU57MKWKP4VI	|A0334811544NRL0EPZY7	|A03765451LCS41DO5OQ2W	|	...	|AZVJ79JI2LIA3	|AZVUBJ1OXBEYU	|AZW7FIUDPE2L8
| --- | --- | --- | --- | --- | --- | --- | --- | 																					
5357956111	| 0 | 0 | 0 | ... | 0 | 0 | 0 |
7535842801	| 0 | 0 | 0 | ... | 0 | 0 | 0 |
9788071198	| 0 | 0 | 0 | ... | 0 | 0 | 0 |
9790775261	| 0 | 0 | 0 | ... | 0 | 0 | 0 |
979078127X	| 0 | 0 | 0 | ... | 0 | 0 | 0 |

Дальнейшие шаги:
1. Сингулярное разложение матрицы(SVD)
2. Составление корреляционной матрицы

7х7 кусочек кор. матрицы:

![image](https://github.com/LisiyLexa/Lab3-Recommendation-System/assets/81087786/6f25e48e-230d-413e-8207-5f220b53063f)

#### Пример использования
Допустим, клиент купил товар B0000535V0.
Выделяем товар с ID B0000535V0 из матрицы
```python
i = "B0000535V0"

product_ids = list(X.index)
product_ID = product_ids.index(i)
product_ID
```
Ищем корреляции для всех товаров с товаром, приобретенным этим клиентом, на основе товаров, оцененных другими покупателями, которые купили тот же товар.
Далее рекомендуем 10 товаров с наивысшей корреляцией.
```python
correlation_product_ID = correlation_matrix[product_ID]
Recommend = list(X.index[correlation_product_ID > 0.90])

# Убираем уже купленный товар
Recommend.remove(i) 

Recommend[0:9]
```
```python terminal
['B00008CMOQ',
 'B000GI3U6C',
 'B000PSTJ4K',
 'B0015KQUO4',
 'B001A6K9NG',
 'B001ET79H8',
 'B0032C6F20',
 'B0037STKW6',
 'B004C1101G']
```

Мы получили топ-10 товаров для отображения рекомендательной системой пользователю, основываясь на истории покупок других пользователей сайта.

## Часть III

### Случай, когда компания впервые создает свой веб-сайт электронной коммерции без какого-либо рейтинга продукта
* Для бизнеса без какой-либо истории покупок может быть иcпользованна рекомендательная система, основанная на поисковых движках. Рекомендации товаров могут быть основаны на текстовом анализе описания продукта

Вид данных:

|product_uid	|product_description|
| --- | --- |
|100001|	Not only do angles make joints stronger, they ...
|100002|	BEHR Premium Textured DECKOVER is an innovativ...
|100003|	Classic architecture meets contemporary design...
|100004|	The Grape Solar 265-Watt Polycrystalline PV So...
|100005|	Update your bathroom with the Delta Vero Singl...

Всего записей: 124428

Векторизуем описание:
```python
vectorizer = TfidfVectorizer(stop_words='english')
X1 = vectorizer.fit_transform(product_descriptions1["product_description"])
```
Кластеризуем товары:
```python
X=X1

kmeans = KMeans(n_clusters = 10, init='k-means++')
y_kmeans = kmeans.fit_predict(X)
```

![image](https://github.com/LisiyLexa/Lab3-Recommendation-System/assets/81087786/57ac2047-672c-46f1-a94b-4e5e7781cc82)

Топ слов в первых двух кластерах:

Cluster 0:
- power
- cutting
- volt
- saw
- tool
- battery
- amp
- lithium
- motor
- m12

Cluster 1:
- water
- toilet
- heater
- warranty
- year
- easy
- tank
- design
- gal
- flush

Предсказываем кластеры, основываясь на поисковом запросе
```python
def show_recommendations(product):
    Y = vectorizer.transform([product])
    prediction = model.predict(Y)
    print_cluster(prediction[0])
```

#### Примеры
**Ключевое слово:** cutting tool
```python
show_recommendations("cutting tool")
```
```python terminal
Cluster 0:
 power
 cutting
 volt
 saw
 tool
 battery
 amp
 lithium
 motor
 m12
```

**Ключевое слово:** spray paint
```python
show_recommendations("spray paint")
```
```python terminal
Cluster 4:
 brush
 roller
 pet
 dust
 easy
 paint
 dog
 ft
 cleaning
 tool
```

**Ключевое слово:** steel drill
```python
show_recommendations("steel drill")
```
```python terminal
Cluster 6:
 metal
 steel
 screw
 screws
 gauge
 drill
 hole
 work
 door
 design
```

Это лучше всего работает, если компания впервые настраивает свой веб-сайт электронной коммерции и у нее изначально нет истории покупок/оценок товаров пользователями. Эта система рекомендаций поможет пользователям получить хорошую рекомендацию в начале, и как только у покупателей появится история покупок, механизм рекомендаций сможет использовать метод коллаборативной фильтрации.
