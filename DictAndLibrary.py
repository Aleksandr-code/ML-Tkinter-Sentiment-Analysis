from textblob import TextBlob
import pandas as pd
import Tokenization
import Lemmatization

def textblob_analyse(text):
    testimonial = TextBlob(text)
    return testimonial.sentiment.polarity

def kartaslovsent_analyse(text):
    tokens = Tokenization.nltk_word_tokenize(text)
    lemma_tokens = Lemmatization.pyMorphy2(tokens)
    print(lemma_tokens)
    df = pd.read_csv('dictionary\kartaslovsent.csv', sep=';')
    df = df[['term', 'value']]
    # print(df[df['term'] == 'абонемент']['value'].tolist()[0])
    sentiment_score = 0
    not_part = False


    for lemma_token in lemma_tokens:
        lemma_token = lemma_token.lower()
        if lemma_token == 'не':
            not_part = True
        value = df[df['term'] == lemma_token]['value'].tolist()
        if value:
            if not_part == True:
                sentiment_score -= value[0]
                not_part = False
            else:
                sentiment_score += value[0]
        else:
            print('Слово:'+lemma_token+' не найдено')
        # df_index = df.index[df['term']=='абонемент'].tolist()
        # print(df_index)
        # print(df[df['term'] == token]['value'].tolist())

    return sentiment_score
