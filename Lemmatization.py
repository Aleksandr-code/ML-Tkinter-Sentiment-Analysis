from nltk.stem import WordNetLemmatizer
import pymorphy2

def wordNetLemmatizer(tokens):
    wordnet_lemmatizer = WordNetLemmatizer()
    lemma_tokens = [wordnet_lemmatizer.lemmatize(w) for w in tokens]
    return lemma_tokens

def pyMorphy2(tokens):
    morph = pymorphy2.MorphAnalyzer()
    lemma_tokens = [morph.parse(w)[0].normal_form for w in tokens]
    return lemma_tokens