import math
import random

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import re
import string


def process_each_news(news_string):
    main_result = []
    set(stopwords.words('english'))
    new_stop_words = stopwords.words('english') + ['I', 'An', 'The', 'She', 'So', 'But', 'but']
    ps = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    result_no_number = re.sub("\d+", "", news_string)
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    result_no_punctuation = result_no_number.translate(translator)
    result_main_words = re.sub(r'\b\w{1,2}\b', '', result_no_punctuation)
    result_to_lower = result_main_words.lower()
    result_split = re.split("\W+", result_to_lower)
    result_no_stop_words = [word for word in result_split if word not in new_stop_words]

    for each_word in result_no_stop_words:
        lemmatize_word = lemmatizer.lemmatize(each_word)
        source_word = ps.stem(lemmatize_word)
        main_result += [source_word]
    return main_result


def find_word_counts_in_news(news_string):
    words_of_news = process_each_news(news_string)
    word_counts_in_news = {}
    for word in words_of_news:
        word_counts_in_news[word] = word_counts_in_news.get(word, 0.0) + 1.0
    return word_counts_in_news


def read_data_file():
    data_file = pd.read_csv(r'data.csv')
    return data_file.dropna()


class DetectCategory:

    def __init__(self):
        self.all_data = {}
        self.train_data = {}
        self.evaluation_data = {}
        self.each_category_count = {}
        self.each_word_counts = {}
        self.class_priors_probability_log = {}
        self.all_words_set = set()
        self.total_data_number = 0
        self.data_file = read_data_file()
        self.correct_detected_travel = 0
        self.correct_detected_business = 0
        self.correct_detected = 0
        self.detected_travel = 0
        self.detected_business = 0
        self.train_data_travel = 0
        self.train_data_business = 0

    def initialize_train_evaluation_data(self):
        self.train_data['TRAVEL'] = []
        self.train_data['BUSINESS'] = []
        self.evaluation_data['TRAVEL'] = []
        self.evaluation_data['BUSINESS'] = []
        self.all_data['TRAVEL'] = []
        self.all_data['BUSINESS'] = []
        travel = 0
        business = 0

        for each_category, each_news in zip(self.data_file['category'], self.data_file['short_description']):
            if each_category == 'TRAVEL':
                travel += 1
                self.all_data['TRAVEL'].append(each_news)
            elif each_category == 'BUSINESS':
                business += 1
                self.all_data['BUSINESS'].append(each_news)

        self.each_category_count['TRAVEL'] = travel
        self.each_category_count['BUSINESS'] = business

        travel_train_data_range = 0
        for each_train_data in self.all_data['TRAVEL']:
            if travel_train_data_range < int(0.8 * self.each_category_count['TRAVEL']):
                self.train_data['TRAVEL'].append(each_train_data)
                travel_train_data_range += 1
            elif travel_train_data_range < self.each_category_count['TRAVEL']:
                self.evaluation_data['TRAVEL'].append(each_train_data)
                travel_train_data_range += 1

        business_train_data_range = 0
        for each_train_data in self.all_data['BUSINESS']:
            if business_train_data_range < int(0.8 * self.each_category_count['BUSINESS']):
                self.train_data['BUSINESS'].append(each_train_data)
                business_train_data_range += 1
            elif business_train_data_range < self.each_category_count['BUSINESS']:
                self.evaluation_data['BUSINESS'].append(each_train_data)
                business_train_data_range += 1

        self.train_data_travel = int(0.8 * self.each_category_count['TRAVEL'])
        self.train_data_business = int(0.8 * self.each_category_count['BUSINESS'])

    def do_over_sampling(self):
        if len(self.train_data['BUSINESS']) < len(self.train_data['TRAVEL']):
            for i in range(len(self.train_data['TRAVEL']) - len(self.train_data['BUSINESS'])):
                self.train_data['BUSINESS'].append(random.choice(self.train_data['BUSINESS']))
                self.train_data_travel = int(len(self.train_data['TRAVEL']))
                self.train_data_business = int(len(self.train_data['TRAVEL']))

        else:
            for i in range(len(self.train_data['BUSINESS']) - len(self.train_data['TRAVEL'])):
                self.train_data['TRAVEL'].append(random.choice(self.train_data['TRAVEL']))
                self.train_data_travel = int(len(self.train_data['BUSINESS']))
                self.train_data_business = int(len(self.train_data['BUSINESS']))

    def get_class_priors_probability(self):
        self.total_data_number = len(self.train_data['TRAVEL']) + len(self.train_data['BUSINESS'])

        self.class_priors_probability_log['TRAVEL'] = \
            math.log(len(self.train_data['TRAVEL']) / self.total_data_number)
        self.class_priors_probability_log['BUSINESS'] = \
            math.log(len(self.train_data['BUSINESS']) / self.total_data_number)

    def get_word_information_each_category(self):
        self.each_word_counts['TRAVEL'] = {}
        self.each_word_counts['BUSINESS'] = {}
        words_counts_in_news = {}

        for each_news in self.train_data['TRAVEL']:
            words_counts_in_news = find_word_counts_in_news(each_news)
            for each_word, word_count in words_counts_in_news.items():
                if each_word not in self.all_words_set:
                    self.all_words_set.add(each_word)
                if each_word not in self.each_word_counts['TRAVEL']:
                    self.each_word_counts['TRAVEL'][each_word] = 0.0
                self.each_word_counts['TRAVEL'][each_word] += word_count

        for each_news in self.train_data['BUSINESS']:
            words_counts_in_news = find_word_counts_in_news(each_news)
            for each_word, word_count in words_counts_in_news.items():
                if each_word not in self.all_words_set:
                    self.all_words_set.add(each_word)
                if each_word not in self.each_word_counts['BUSINESS']:
                    self.each_word_counts['BUSINESS'][each_word] = 0.0
                self.each_word_counts['BUSINESS'][each_word] += word_count

    def calculate_probability(self):

        self.get_class_priors_probability()

        for news in self.evaluation_data['BUSINESS']:
            counts = find_word_counts_in_news(news)
            travel_prob2 = 0
            business_prob2 = 0
            for word, _ in counts.items():
                if word not in self.all_words_set:
                    continue

                log_likelihood_travel = math.log(
                    (self.each_word_counts['TRAVEL'].get(word, 0.0) + 1) / (
                            self.train_data_travel + len(self.all_words_set)))
                log_likelihood_business = math.log(
                    (self.each_word_counts['BUSINESS'].get(word, 0.0) + 1) / (
                            self.train_data_business+ len(self.all_words_set)))

                travel_prob2 += log_likelihood_travel
                business_prob2 += log_likelihood_business

            travel_prob2 += self.class_priors_probability_log['TRAVEL']
            business_prob2 += self.class_priors_probability_log['BUSINESS']

            if business_prob2 > travel_prob2:
                self.correct_detected_business += 1
                self.detected_business += 1
            else:
                self.detected_travel += 1

        for news in self.evaluation_data['TRAVEL']:
            counts = find_word_counts_in_news(news)
            travel_prob = 0
            business_prob = 0
            for word, _ in counts.items():
                if word not in self.all_words_set:
                    continue

                log_likelihood_travel = math.log(
                    (self.each_word_counts['TRAVEL'].get(word, 0.0) + 1) / (
                            self.train_data_travel + len(self.all_words_set)))
                log_likelihood_business = math.log(
                    (self.each_word_counts['BUSINESS'].get(word, 0.0) + 1) / (
                            self.train_data_business + len(self.all_words_set)))

                travel_prob += log_likelihood_travel
                business_prob += log_likelihood_business

            travel_prob += self.class_priors_probability_log['TRAVEL']
            business_prob += self.class_priors_probability_log['BUSINESS']

            if travel_prob > business_prob:
                self.correct_detected_travel += 1
                self.detected_travel += 1
            else:
                self.detected_business += 1

    def calculate_recall_precision_accuracy(self):
        travel_recall = (self.correct_detected_travel / len(self.evaluation_data['TRAVEL'])) * 100
        travel_precision = (self.correct_detected_travel / self.detected_travel) * 100

        business_recall = (self.correct_detected_business / len(self.evaluation_data['BUSINESS'])) * 100
        business_precision = (self.correct_detected_business / self.detected_business) * 100

        accuracy = 100 * ((self.correct_detected_travel + self.correct_detected_business) / (
                    len(self.evaluation_data['TRAVEL']) + len(self.evaluation_data['BUSINESS'])))
        print("travel_recall = " + str(travel_recall))
        print("travel_precision = " + str(travel_precision))
        print("********************************************")
        print("business_recall = " + str(business_recall))
        print("business_precision = " + str(business_precision))
        print("accuracy = " + str(accuracy))
        print("\n")
        print("Travel Evaluation Data Number = " + str(len(self.evaluation_data['TRAVEL'])))
        print("Business Evaluation Data Number = " + str(len(self.evaluation_data['BUSINESS'])))

        print("Check")
        print(self.correct_detected_business)
        print(self.detected_business)


d = DetectCategory()
d.initialize_train_evaluation_data()
d.do_over_sampling()
d.get_word_information_each_category()
d.calculate_probability()
d.calculate_recall_precision_accuracy()
