import nltk
import re

from nltk.tokenize import word_tokenize
from nltk.tokenize import WordPunctTokenizer
from nltk.tokenize import TweetTokenizer
# from razdel import tokenize
import razdel

def check_str_function(text):
    if (type(text) != 'str'):
        text = str(text)
        return text
    else:
        return text 

def split_method(text):
    text = check_str_function(text)
    return text.split()

def nltk_word_tokenize(text):
    text = check_str_function(text)
    return word_tokenize(text)

def nltk_WordPunctTokenizer(text):
    text = check_str_function(text)
    tk = WordPunctTokenizer()
    return tk.tokenize(text)

def razdel_tokenizer(text):
    text = check_str_function(text)
    return [_.text for _ in list(razdel.tokenize(text))]

def nltk_TweetTokenizer(text):
    text = check_str_function(text)
    tk = TweetTokenizer()
    return tk.tokenize(text)

# def spacy_tokenize(text):
#     sp = spacy.load('ru_core_news_sm')
#     doc = sp.tokenizer(text)
#     return [token.text for token in doc]

def re_tokenize(text):
    text = check_str_function(text)
    pattern = re.compile(r'([^\W\d]+|\d+|[^\w\s])')
    text = pattern.findall(text)
    return text
