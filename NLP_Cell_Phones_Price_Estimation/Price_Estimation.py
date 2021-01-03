from __future__ import unicode_literals

import nltk
import pandas as pd
import hazm
from hazm import *
from persiantools import characters
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 1000)


def preprocess_on_each_sent(sentence, _lemmatizer):
    new_sentence = ""
    punctuations = ['.', '؟', '/', '،', '!']
    stop_words_list = hazm.utils.stopwords_list()
    list_of_words = word_tokenize(sentence)
    for i in list_of_words:
        if i not in stop_words_list and i not in punctuations:
            new_word = _lemmatizer.lemmatize(i)
            new_sentence = new_sentence + " " + new_word
    return new_sentence


def read_input_file():
    data_file = pd.read_csv(r'mobile_phone_dataset.csv')
    return data_file

def process_data_file(input_data_file):
    days = []
    time = []
    lemmatizer = Lemmatizer()
    stemmer = Stemmer()
    normalizer = Normalizer()
    stemmer = Stemmer()
    input_data_file = input_data_file[input_data_file.price >= 0].reset_index(drop=True)
    input_data_file['desc'] = input_data_file.apply(lambda each_row: normalizer.normalize(each_row['desc']), axis=1)
    input_data_file['title'] = input_data_file.apply(lambda each_row: normalizer.normalize(each_row['title']), axis=1)
    input_data_file['desc'] = input_data_file.apply(lambda each_row: preprocess_on_each_sent(each_row['desc'], lemmatizer), axis=1)
    input_data_file['title'] = input_data_file.apply(lambda each_row: preprocess_on_each_sent(each_row['title'], lemmatizer), axis=1)



    input_data_file['created_at'] = df.apply(lambda each_row: nltk.word_tokenize(each_row['created_at']), axis=1)
    input_data_file.apply(lambda each_row: days.append(each_row['created_at'][0]), axis=1)
    input_data_file.apply(lambda each_row: time.append(each_row['created_at'][1]), axis=1)
    input_data_file['created_at_day'] = days
    input_data_file['created_at_time'] = time
    input_data_file = input_data_file.drop('created_at', axis=1)

    input_data_file['brand'] = df.apply(lambda each_row: each_row['brand'].split('::')[0], axis=1)

    return input_data_file


def split_train_test_data(data_file):
    features = data_file.drop('price', axis=1)
    target = data_file['price']
    features_train, features_test, target_train, target_test = train_test_split(features, target, train_size=0.8)
    return features_train, features_test, target_train, target_test


def manage_data(features_train, features_test, data_file):
    vectorizer = CountVectorizer(lowercase=False, binary=True)
    #df_brand_oneHot = vectorizer.fit_transform(data_file['brand'].values)
    train_brand_oneHot = vectorizer.fit_transform(features_train['brand'].values)
    test_brand_oneHot = vectorizer.transform(features_test['brand'].values)

    #df_city_oneHot = vectorizer.fit_transform(data_file['city'].values)
    train_city_oneHot = vectorizer.fit_transform(features_train['city'].values)
    test_city_oneHot = vectorizer.transform(features_test['city'].values)

    #df_day_oneHot = vectorizer.fit_transform(data_file['created_at_day'].values)
    train_day_oneHot = vectorizer.fit_transform(features_train['created_at_day'].values)
    test_day_oneHot = vectorizer.transform(features_test['created_at_day'].values)

    #df_time_oneHot = vectorizer.fit_transform(data_file['created_at_time'].values)
    train_time_oneHot = vectorizer.fit_transform(features_train['created_at_time'].values)
    test_time_oneHot = vectorizer.transform(features_test['created_at_time'].values)

    df_image_count = [str(i) for i in data_file['image_count'].values]
    train_image_count = [str(i) for i in features_train['image_count'].values]
    test_image_count = [str(i) for i in features_test['image_count'].values]

    print(train_image_count)
    df_image_count_oneHot = vectorizer.fit_transform(df_image_count)
    train_image_count_oneHot = vectorizer.transform(train_image_count)
    test_image_count_oneHot = vectorizer.transform(test_image_count)

    vectorizer = TfidfVectorizer(ngram_range=(1, 3), min_df=3, max_features=250)
    #df_title_tfidf = vectorizer.fit_transform(data_file['title'].values)
    train_title_tfidf = vectorizer.fit_transform(features_train['title'].values)
    test_title_tfidf = vectorizer.transform(features_test['title'].values)

    vectorizer = TfidfVectorizer(ngram_range=(1, 3), min_df=5, max_features=500)
    #df_desc_tfidf = vectorizer.fit_transform(data_file['desc'].values)
    train_desc_tfidf = vectorizer.fit_transform(features_train['desc'].values)
    test_desc_tfidf = vectorizer.transform(features_test['desc'].values)

    train_sparse = hstack((
        train_brand_oneHot, train_city_oneHot, train_day_oneHot, train_time_oneHot, train_image_count_oneHot,
        train_title_tfidf,
        train_desc_tfidf)).tocsr()
    test_sparse = hstack((test_brand_oneHot, test_city_oneHot, test_day_oneHot, test_time_oneHot,
                          test_image_count_oneHot, test_title_tfidf,
                          test_desc_tfidf)).tocsr()

    return train_sparse, test_sparse


def model_prediction(pro_features_train, pro_features_test, target_train, target_test):
    lgr = LogisticRegression(random_state=0, max_iter=10000)
    lgr.fit(pro_features_train, target_train)

    actual_train_data = list(target_train)
    predict_train_data = lgr.predict(pro_features_train)
    actual_test_data = list(target_test)
    predict_test_data = lgr.predict(pro_features_test)

    acc_test = lgr.score(pro_features_test, actual_test_data)
    acc_train = lgr.score(pro_features_train, actual_train_data)
    print('Accuracy Test= %.3f' % (acc_test * 100))
    print('Accuracy Train= %.3f' % (acc_train * 100))

def model2_prediction(pro_features_train, pro_features_test, target_train, target_test):
    trg = DecisionTreeRegressor(random_state=0, max_depth = 20)
    trg.fit(pro_features_train, target_train)

    actual_train_data = list(target_train)
    predict_train_data = trg.predict(pro_features_train)
    actual_test_data = list(target_test)
    predict_test_data = trg.predict(pro_features_test)

    acc_test = trg.score(pro_features_test, actual_test_data)
    acc_train = trg.score(pro_features_train, actual_train_data)
    print('Accuracy Test= %.3f' % (acc_test * 100))
    print('Accuracy Train= %.3f' % (acc_train * 100))


df = read_input_file()
input_data = process_data_file(df)
print(input_data)
main_features_train, main_features_test, main_target_train, main_target_test = split_train_test_data(input_data)
processed_features_train, processed_features_test = manage_data(main_features_train, main_features_test, input_data)
#model_prediction(processed_features_train, processed_features_test, main_target_train, main_target_test)
model2_prediction(processed_features_train, processed_features_test, main_target_train, main_target_test)
