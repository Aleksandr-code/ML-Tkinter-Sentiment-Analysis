from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

# Переводим метки сентимент-анализа в бинарное представление
def defineLabels(sentiment_target_data):
    #Создаем бинарные метки с помощью LabelBinarizer() из sklearn 
    lb=LabelBinarizer()
    #трансформирем сентимент-данные в бинарное представление
    sentiment_data=lb.fit_transform(sentiment_target_data)
    # print(sentiment_data.shape)
    return sentiment_data

# Из пятизвездочной системы в бинарное представление
def get_sentiment_five_label(score):
    if score in [4.0, 5.0]:
        return 1
    elif score in [1.0, 2.0, 3.0]:
        return 0
    
# Из многозвездочной системы в бинарное представление
def get_sentiment_many_label(score):
    if score >= 0:
        return 1
    elif score < 0:
        return 0

# Разделяем тренировачные и тестовые данные
def splitDataset(text_data, sentiment_data):

    norm_train_data, norm_test_data, train_sentiments, test_sentiments = train_test_split(text_data, sentiment_data, 
                                                    train_size=0.75, 
                                                    random_state=42,
                                                    stratify=sentiment_data)
    return norm_train_data, norm_test_data, train_sentiments, test_sentiments