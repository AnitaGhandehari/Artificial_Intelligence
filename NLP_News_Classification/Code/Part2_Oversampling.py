import math
import random
import csv

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
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
        lemmatize_word = lemmatizer.lemmatize(each_word, 'n')
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
        self.correct_detected_beauty = 0
        self.correct_detected = 0
        self.detected_travel = 0
        self.detected_business = 0
        self.detected_beauty = 0
        self.train_data_travel = 0
        self.train_data_business = 0
        self.train_data_beauty = 0
        self.actual_data = []
        self.predict_data = []

    def initialize_train_evaluation_data(self):
        self.train_data['TRAVEL'] = []
        self.train_data['BUSINESS'] = []
        self.train_data['STYLE & BEAUTY'] = []
        self.evaluation_data['TRAVEL'] = []
        self.evaluation_data['BUSINESS'] = []
        self.evaluation_data['STYLE & BEAUTY'] = []
        self.all_data['TRAVEL'] = []
        self.all_data['BUSINESS'] = []
        self.all_data['STYLE & BEAUTY'] = []
        travel = 0
        business = 0
        beauty = 0

        for each_category, each_news in zip(self.data_file['category'], self.data_file['short_description']):
            if each_category == 'TRAVEL':
                travel += 1
                self.all_data['TRAVEL'].append(each_news)
            elif each_category == 'BUSINESS':
                business += 1
                self.all_data['BUSINESS'].append(each_news)
            elif each_category == 'STYLE & BEAUTY':
                beauty += 1
                self.all_data['STYLE & BEAUTY'].append(each_news)

        self.each_category_count['TRAVEL'] = travel
        self.each_category_count['BUSINESS'] = business
        self.each_category_count['STYLE & BEAUTY'] = beauty

        travel_train_data_range = 0
        for each_train_data in self.all_data['TRAVEL']:
            if travel_train_data_range < int(0.8 * self.each_category_count['TRAVEL']):
                self.train_data['TRAVEL'].append(each_train_data)
                travel_train_data_range += 1
            elif travel_train_data_range < self.each_category_count['TRAVEL']:
                self.evaluation_data['TRAVEL'].append(each_train_data)
                travel_train_data_range += 1

        business_train_data_range = 0
        for each_train_data2 in self.all_data['BUSINESS']:
            if business_train_data_range < int(0.8 * self.each_category_count['BUSINESS']):
                self.train_data['BUSINESS'].append(each_train_data2)
                business_train_data_range += 1
            elif business_train_data_range < self.each_category_count['BUSINESS']:
                self.evaluation_data['BUSINESS'].append(each_train_data2)
                business_train_data_range += 1

        beauty_train_data_range = 0
        for each_train_data3 in self.all_data['STYLE & BEAUTY']:
            if beauty_train_data_range < int(0.8 * self.each_category_count['STYLE & BEAUTY']):
                self.train_data['STYLE & BEAUTY'].append(each_train_data3)
                beauty_train_data_range += 1
            elif beauty_train_data_range < self.each_category_count['STYLE & BEAUTY']:
                self.evaluation_data['STYLE & BEAUTY'].append(each_train_data3)
                beauty_train_data_range += 1

        self.train_data_travel = int(0.8 * self.each_category_count['TRAVEL'])
        self.train_data_business = int(0.8 * self.each_category_count['BUSINESS'])
        self.train_data_beauty = int(0.8 * self.each_category_count['STYLE & BEAUTY'])

    def do_oversampling(self):
        if len(self.train_data['TRAVEL']) > len(self.train_data['BUSINESS']) and \
                len(self.train_data['TRAVEL']) > len(self.train_data['STYLE & BEAUTY']):

            for i in range(len(self.train_data['TRAVEL']) - len(self.train_data['BUSINESS'])):
                self.train_data['BUSINESS'].append(random.choice(self.train_data['BUSINESS']))

            for i in range(len(self.train_data['TRAVEL']) - len(self.train_data['STYLE & BEAUTY'])):
                self.train_data['STYLE & BEAUTY'].append(random.choice(self.train_data['STYLE & BEAUTY']))

            self.train_data_travel = len(self.train_data['TRAVEL'])
            self.train_data_business = len(self.train_data['TRAVEL'])
            self.train_data_beauty = len(self.train_data['TRAVEL'])

        elif len(self.train_data['BUSINESS']) > len(self.train_data['TRAVEL']) and \
                len(self.train_data['BUSINESS']) > len(self.train_data['STYLE & BEAUTY']):

            for i in range(len(self.train_data['BUSINESS']) - len(self.train_data['TRAVEL'])):
                self.train_data['TRAVEL'].append(random.choice(self.train_data['TRAVEL']))

            for i in range(len(self.train_data['BUSINESS']) - len(self.train_data['TRAVEL'])):
                self.train_data['STYLE & BEAUTY'].append(random.choice(self.train_data['STYLE & BEAUTY']))

            self.train_data_travel = len(self.train_data['BUSINESS'])
            self.train_data_business = len(self.train_data['BUSINESS'])
            self.train_data_beauty = len(self.train_data['BUSINESS'])

        elif len(self.train_data['STYLE & BEAUTY']) > len(self.train_data['TRAVEL']) and \
                len(self.train_data['STYLE & BEAUTY']) > len(self.train_data['BUSINESS']):

            for i in range(len(self.train_data['STYLE & BEAUTY']) - len(self.train_data['TRAVEL'])):
                self.train_data['TRAVEL'].append(random.choice(self.train_data['TRAVEL']))

            for i in range(len(self.train_data['STYLE & BEAUTY']) - len(self.train_data['BUSINESS'])):
                self.train_data['BUSINESS'].append(random.choice(self.train_data['BUSINESS']))

            self.train_data_travel = len(self.train_data['STYLE & BEAUTY'])
            self.train_data_business = len(self.train_data['STYLE & BEAUTY'])
            self.train_data_beauty = len(self.train_data['STYLE & BEAUTY'])

    def get_class_priors_probability(self):
        self.total_data_number = (len(self.train_data['TRAVEL']) + len(self.train_data['BUSINESS']) + len(
            self.train_data['STYLE & BEAUTY']))
        self.class_priors_probability_log['TRAVEL'] = math.log(
            len(self.train_data['TRAVEL']) / self.total_data_number)
        self.class_priors_probability_log['BUSINESS'] = math.log(
            len(self.train_data['BUSINESS']) / self.total_data_number)
        self.class_priors_probability_log['STYLE & BEAUTY'] = math.log(
            len(self.train_data['STYLE & BEAUTY']) / self.total_data_number)

    def get_word_information_each_category(self):
        self.each_word_counts['TRAVEL'] = {}
        self.each_word_counts['BUSINESS'] = {}
        self.each_word_counts['STYLE & BEAUTY'] = {}
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

        for each_news in self.train_data['STYLE & BEAUTY']:
            words_counts_in_news = find_word_counts_in_news(each_news)
            for each_word, word_count in words_counts_in_news.items():
                if each_word not in self.all_words_set:
                    self.all_words_set.add(each_word)
                if each_word not in self.each_word_counts['STYLE & BEAUTY']:
                    self.each_word_counts['STYLE & BEAUTY'][each_word] = 0.0
                self.each_word_counts['STYLE & BEAUTY'][each_word] += word_count

    def calculate_probability(self):
        self.get_class_priors_probability()
        for news in self.evaluation_data['TRAVEL']:
            self.actual_data.append('TRAVEL')
            counts = find_word_counts_in_news(news)
            travel_prob = 0
            business_prob = 0
            beauty_prob = 0
            for word, _ in counts.items():
                if word not in self.all_words_set:
                    continue

                log_likelihood_travel = math.log(
                    (self.each_word_counts['TRAVEL'].get(word, 0.0) + 1) / (
                            self.train_data_travel + len(self.all_words_set)))
                log_likelihood_business = math.log(
                    (self.each_word_counts['BUSINESS'].get(word, 0.0) + 1) / (
                            self.train_data_business + len(self.all_words_set)))
                log_likelihood_beauty = math.log(
                    (self.each_word_counts['STYLE & BEAUTY'].get(word, 0.0) + 1) / (
                            self.train_data_beauty + len(self.all_words_set)))

                travel_prob += log_likelihood_travel
                business_prob += log_likelihood_business
                beauty_prob += log_likelihood_beauty

            travel_prob += self.class_priors_probability_log['TRAVEL']
            business_prob += self.class_priors_probability_log['BUSINESS']
            beauty_prob += self.class_priors_probability_log['STYLE & BEAUTY']

            if business_prob > travel_prob and business_prob > beauty_prob:
                self.detected_business += 1
                self.predict_data.append('BUSINESS')
            elif travel_prob >= business_prob and travel_prob >= beauty_prob:
                self.correct_detected_travel += 1
                self.detected_travel += 1
                self.predict_data.append('TRAVEL')
            elif beauty_prob > business_prob and beauty_prob > travel_prob:
                self.detected_beauty += 1
                self.predict_data.append('STYLE & BEAUTY')

        for news in self.evaluation_data['BUSINESS']:
            self.actual_data.append('BUSINESS')
            counts = find_word_counts_in_news(news)
            travel_prob2 = 0
            business_prob2 = 0
            beauty_prob2 = 0
            for word, _ in counts.items():
                if word not in self.all_words_set:
                    continue

                log_likelihood_travel = math.log(
                    (self.each_word_counts['TRAVEL'].get(word, 0.0) + 1) / (
                            self.train_data_travel + len(self.all_words_set)))
                log_likelihood_business = math.log(
                    (self.each_word_counts['BUSINESS'].get(word, 0.0) + 1) / (
                            self.train_data_business + len(self.all_words_set)))
                log_likelihood_beauty = math.log(
                    (self.each_word_counts['STYLE & BEAUTY'].get(word, 0.0) + 1) / (
                            self.train_data_beauty + len(self.all_words_set)))

                travel_prob2 += log_likelihood_travel
                business_prob2 += log_likelihood_business
                beauty_prob2 += log_likelihood_beauty

            travel_prob2 += self.class_priors_probability_log['TRAVEL']
            business_prob2 += self.class_priors_probability_log['BUSINESS']
            beauty_prob2 += self.class_priors_probability_log['STYLE & BEAUTY']

            if business_prob2 >= travel_prob2 and business_prob2 >= beauty_prob2:
                self.correct_detected_business += 1
                self.detected_business += 1
                self.predict_data.append('BUSINESS')
            elif travel_prob2 > business_prob2 and travel_prob2 > beauty_prob2:
                self.detected_travel += 1
                self.predict_data.append('TRAVEL')
            elif beauty_prob2 > business_prob2 and beauty_prob2 > travel_prob2:
                self.detected_beauty += 1
                self.predict_data.append('STYLE & BEAUTY')

        for news in self.evaluation_data['STYLE & BEAUTY']:
            self.actual_data.append('STYLE & BEAUTY')
            counts = find_word_counts_in_news(news)
            travel_prob2 = 0
            business_prob2 = 0
            beauty_prob2 = 0
            for word, _ in counts.items():
                if word not in self.all_words_set:
                    continue

                log_likelihood_travel = math.log(
                    (self.each_word_counts['TRAVEL'].get(word, 0.0) + 1) / (
                            self.train_data_travel + len(self.all_words_set)))
                log_likelihood_business = math.log(
                    (self.each_word_counts['BUSINESS'].get(word, 0.0) + 1) / (
                            self.train_data_business + len(self.all_words_set)))
                log_likelihood_beauty = math.log(
                    (self.each_word_counts['STYLE & BEAUTY'].get(word, 0.0) + 1) / (
                            self.train_data_beauty + len(self.all_words_set)))

                travel_prob2 += log_likelihood_travel
                business_prob2 += log_likelihood_business
                beauty_prob2 += log_likelihood_beauty

            travel_prob2 += self.class_priors_probability_log['TRAVEL']
            business_prob2 += self.class_priors_probability_log['BUSINESS']
            beauty_prob2 += self.class_priors_probability_log['STYLE & BEAUTY']

            if business_prob2 > travel_prob2 and business_prob2 > beauty_prob2:
                self.detected_business += 1
                self.predict_data.append('BUSINESS')
            elif travel_prob2 > business_prob2 and travel_prob2 > beauty_prob2:
                self.detected_travel += 1
                self.predict_data.append('TRAVEL')
            elif beauty_prob2 >= business_prob2 and beauty_prob2 >= travel_prob2:
                self.correct_detected_beauty += 1
                self.detected_beauty += 1
                self.predict_data.append('STYLE & BEAUTY')

    def calculate_recall_precision_accuracy(self):
        travel_recall = (self.correct_detected_travel / len(self.evaluation_data['TRAVEL'])) * 100
        travel_precision = (self.correct_detected_travel / self.detected_travel) * 100

        business_recall = (self.correct_detected_business / len(self.evaluation_data['BUSINESS'])) * 100
        business_precision = (self.correct_detected_business / self.detected_business) * 100

        beauty_recall = (self.correct_detected_beauty / len(self.evaluation_data['STYLE & BEAUTY'])) * 100
        beauty_precision = (self.correct_detected_beauty / self.detected_beauty) * 100

        accuracy = 100 * (
                (self.correct_detected_travel + self.correct_detected_business + self.correct_detected_beauty) /
                (len(self.evaluation_data['TRAVEL']) + len(self.evaluation_data['BUSINESS']) + len(
                    self.evaluation_data['STYLE & BEAUTY'])))

        print("travel_recall = " + str(travel_recall))
        print("travel_precision = " + str(travel_precision))
        print("********************************************")
        print("business_recall = " + str(business_recall))
        print("business_precision = " + str(business_precision))
        print("********************************************")
        print("beauty_recall = " + str(beauty_recall))
        print("beauty_precision = " + str(beauty_precision))
        print("accuracy = " + str(accuracy))
        print("\n")
        print("Travel Evaluation Data Number = " + str(len(self.evaluation_data['TRAVEL'])))
        print("Business Evaluation Data Number = " + str(len(self.evaluation_data['BUSINESS'])))

        matrix = confusion_matrix(self.actual_data, self.predict_data)
        report = classification_report(self.actual_data, self.predict_data)
        print("confusion_matrix")
        print(matrix)
        print("Report")
        print(report)

    def analise_test_data(self):
        output_data = []
        test_file = pd.read_csv(r'test.csv')
        middle = pd.DataFrame(zip(test_file['index'], test_file['short_description']), columns=['index','short_description'])
        test_file_no_null = middle.dropna()
        self.get_class_priors_probability()
        print(middle)
        for news, index in zip(test_file_no_null['short_description'], test_file_no_null['index']):
            travel_prob = 0
            business_prob = 0
            beauty_prob = 0
            counts = find_word_counts_in_news(news)
            for word, _ in counts.items():
                if word not in self.all_words_set:
                    continue

                log_likelihood_travel = math.log(
                    (self.each_word_counts['TRAVEL'].get(word, 0.0) + 1) / (
                            self.train_data_travel + len(self.all_words_set)))
                log_likelihood_business = math.log(
                    (self.each_word_counts['BUSINESS'].get(word, 0.0) + 1) / (
                            self.train_data_business + len(self.all_words_set)))
                log_likelihood_beauty = math.log(
                    (self.each_word_counts['STYLE & BEAUTY'].get(word, 0.0) + 1) / (
                            self.train_data_beauty + len(self.all_words_set)))

                travel_prob += log_likelihood_travel
                business_prob += log_likelihood_business
                beauty_prob += log_likelihood_beauty

            travel_prob += self.class_priors_probability_log['TRAVEL']
            business_prob += self.class_priors_probability_log['BUSINESS']
            beauty_prob += self.class_priors_probability_log['STYLE & BEAUTY']

            if business_prob > travel_prob and business_prob > beauty_prob:
                output_data.append([index, "BUSINESS"])

            elif travel_prob >= business_prob and travel_prob >= beauty_prob:
                output_data.append([index, "TRAVEL"])

            elif beauty_prob > business_prob and beauty_prob > travel_prob:
                output_data.append([index, "STYLE & BEAUTY"])


        output = pd.DataFrame(output_data, columns=['index', 'category'])
        output.to_csv(path_or_buf='output.csv',  index=False)


d = DetectCategory()
d.initialize_train_evaluation_data()
d.do_oversampling()
d.get_word_information_each_category()
d.calculate_probability()
d.calculate_recall_precision_accuracy()
d.analise_test_data()
