import re
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

import Tokenization

#Удаление html разметки
def strip_html(text):
    soup = BeautifulSoup(str(text), "html.parser")
    return soup.get_text()

#Удаление скобок
def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', str(text))

#Удаление специальных символов - русский
def remove_special_characters_ru(text, remove_digits=True):
    pattern=r'[^а-яА-я0-9\s]'
    text=re.sub(pattern,'',str(text))
    return text

#Удаление специальных символов - английский
def remove_special_characters_en(text, remove_digits=True):
    pattern=r'[^a-zA-z0-9\s]'
    text=re.sub(pattern,'',str(text))
    return text

#Удаление стоп-слов (nltk-english)
def remove_stopwords(tokens, is_lower_case=False):
    stopword_list=nltk.corpus.stopwords.words('english')
    # stop=set(stopwords.words('english'))
    # print(stop)

    # tokens = Tokenization.nltk_word_tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
        return filtered_tokens
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
        return filtered_tokens

#Создание текста из токенов после всех операций   
def filtered_text(tokens):
    filtered_text = ' '.join(tokens)    
    return filtered_text