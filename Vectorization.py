
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

#Векторизация с помощью Bag of Words
def vectorize_BoW(norm_train_data, norm_test_data):
    cv=CountVectorizer(min_df=0,max_df=1,binary=False,ngram_range=(1,3))
    #трансформация тренировачных данных
    cv_train_data=cv.fit_transform(norm_train_data)
    #трансформация тестовых данных
    cv_test_data=cv.transform(norm_test_data)
    return cv, cv_train_data, cv_test_data

def vectorize_Tfidf(norm_train_data, norm_test_data):
    # Векторизация с помощью Tfidf
    tv=TfidfVectorizer(min_df=0,max_df=1,use_idf=True,ngram_range=(1,3))
    #трансформация тренировачных данных
    tv_train_data=tv.fit_transform(norm_train_data)
    #трансформация тестовых данных
    tv_test_data=tv.transform(norm_test_data)
    return tv, tv_train_data,tv_test_data

# def vectorize_predict(text):
#     tv=TfidfVectorizer(min_df=0,max_df=1,use_idf=True,ngram_range=(1,3))
#     tv_text=tv.transform([text])
#     return tv_text

