## About Project

<p>Sentiment Analysis with Python/ML</p>
<p>Приложение - Конфигуратор моделей для анализа тональности текста</p>

<p align="center"><img src="https://github.com/Aleksandr-code/ML-Tkinter-Sentiment-Analysis/raw/master/images/screenshot.jpg" width="400" alt="screenshot"></p>

Функционал:
- Загрузка файла (*.csv, *.tsv)
- Токенизация:
    - split;
    - word_tokenize; 
    - WordPunctTokenizer; 
    - Razdel, TweetTokenizer;
    - Re.findall(\w+|\d+|\p+);
- Нормализация данных:
    - Удаление html разметки
    - Удаление квадратных скобок
    - Удаление специальных символов "russian"
    - Удаление специальных символов "english"
    - Удаление стоп-слов "english"
- Cтемминг:
    - Стеммер Портера
    - Стеммер SnowBall
    - Стеммер Ланкастера
- Лемматизация:
    - Лемматизатор WordNet
    - Лемматизатор pyMorphy2
- Векторизация:
    - Bag of words
    - TF-IDF
- Метод машинного обучения:
    - Логистическая регрессия
    - Наивный байесовский классификатор
    - Cтохастический градиентный спуск
    - Метод k-ближайших соседей
    - Метод опорных векторов
    - Деревья решений
    - Случайный лес
    - Градиентный бустинг
    - Многослойный перцептрон
- Классификация текста на английском языке с помощью библиотеки TextBlob (подход на основе правил)
- Классификация текста на русском языке с помощью словаря КартаСловСент (подход на основе словаря)



## Stack

Python, Tkinter, Pandas, Sklearn.

## Installation

- <code>pip install -r requirements.txt</code>

## Launch

- <code>python main.py</code>
