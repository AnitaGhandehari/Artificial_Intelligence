import random

import imblearn as imblearn
import pandas as pd
import sklearn
from sklearn import preprocessing
from sklearn.ensemble import BaggingClassifier, VotingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import classification_report, precision_score, accuracy_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def process_input_data(file_name):
    data_file = pd.read_csv(file_name)
    data_file = data_file[data_file['Total Quantity'] >= 0]
    data_file = data_file[data_file['Total Price'] >= 0]
    data_file = data_file[data_file['Purchase Count'] >= 0]
    features = data_file.drop('Is Back', axis=1)
    features = features.drop('Unnamed: 0', axis=1)
    features = features.drop('Customer ID', axis=1)
    target = data_file['Is Back']
    countries = features['Country']
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(features.Country)
    features['Country'] = label_encoder.transform(features['Country'])

    #one_hot = OneHotEncoder()
    #one_hot.fit(list(zip(countries, features['Country'])))
    #features['Country'] = one_hot.transform(list(zip(countries, features['Country']))).toarray()


    features['Date'] = features['Date'].str.split('-').str[0] + features['Date'].str.split('-').str[1] + \
                       features['Date'].str.split('-').str[2]
    features['Date'] = pd.to_numeric(features['Date'])

    scaler = StandardScaler(with_std=True)
    new = list(zip(features['Total Price'],  features['Purchase Count']))
    scaler.fit(new)
    new = scaler.transform(new)

    features['Total Price'] = new[:, 0]
    features['Purchase Count'] = new[:, 1]


    return features, target


def get_train_test_data(_features, _target):
    features_train, features_test, target_train, target_test = train_test_split(_features, _target, train_size=0.8)
    return features_train, features_test, target_train, target_test


def get_information_gain(_train_features, _train_target):
    feature_labels = ['Total Quantity', 'Total Price', 'Country', 'Date', 'Purchase Count']

    information_gain_result = sklearn.feature_selection.mutual_info_classif(_train_features, _train_target, copy=True)
    print(information_gain_result)

    left = [1, 2, 3, 4, 5]
    plt.bar(left, information_gain_result, tick_label=feature_labels, width=0.6, color=['green', 'red'])
    plt.xlabel('Features')
    plt.ylabel('Gain')
    plt.title('Information Gain')
    plt.show()

def classify_by_KNN(_train_features, _train_target, _test_features, _test_target):
    accuracy_train_values = []
    accuracy_test_values = []

    knn_classifier = KNeighborsClassifier(n_neighbors=3)
    knn_classifier.fit(_train_features, _train_target)

    actual_train_data = list(_train_target)
    predict_train_data = knn_classifier.predict(_train_features)
    actual_test_data = list(_test_target)
    predict_test_data = knn_classifier.predict(_test_features)

    for i in range(1, 100):
        knn_classifier = KNeighborsClassifier(n_neighbors=i).fit(_train_features, _train_target)

        accuracy_train_values.append(
            sklearn.metrics.accuracy_score(actual_train_data, knn_classifier.predict(_train_features)))
        accuracy_test_values.append(
            sklearn.metrics.accuracy_score(actual_test_data, knn_classifier.predict(_test_features)))

    x = list(range(100))
    plt.plot(accuracy_train_values)
    plt.plot(accuracy_test_values)
    plt.legend(["Train", "Test"])
    plt.xlabel('n_neighbors')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Diagram for KNN Classifier')
    plt.show()

    train_report = classification_report(actual_train_data, predict_train_data)
    test_report = classification_report(actual_test_data, predict_test_data)

    print('train_report')
    print(train_report)
    print('test_report')
    print(test_report)


def classify_by_tree(_train_features, _train_target, _test_features, _test_target):
    precision_train_values = []
    precision_test_values = []

    tree_classifier = tree.DecisionTreeClassifier(max_depth=4)
    tree_classifier = tree_classifier.fit(_train_features, _train_target)

    actual_train_data = list(_train_target)
    predict_train_data = tree_classifier.predict(_train_features)
    actual_test_data = list(_test_target)
    predict_test_data = tree_classifier.predict(_test_features)

    for i in range(1, 20):
        tree_classifier = tree.DecisionTreeClassifier(max_depth=i).fit(_train_features, _train_target)

        precision_train_values.append(
            sklearn.metrics.accuracy_score(tree_classifier.predict(_train_features), actual_train_data))
        precision_test_values.append(
            sklearn.metrics.accuracy_score(tree_classifier.predict(_test_features), actual_test_data))

    x = list(range(100))
    plt.plot(precision_train_values)
    plt.plot(precision_test_values)
    plt.legend(["Train", "Test"])
    plt.xlabel('max_depth')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Diagram for Tree Classifier')
    plt.show()

    train_report = classification_report(actual_train_data, predict_train_data)
    test_report = classification_report(actual_test_data, predict_test_data)

    print('train_report')
    print(train_report)
    print('test_report')
    print(test_report)


def classify_by_logistic(_train_features, _train_target, _test_features, _test_target):

    logistic_classifier = LogisticRegression()




    logistic_classifier = logistic_classifier.fit(_train_features, _train_target)

    actual_train_data = list(_train_target)
    predict_train_data = logistic_classifier.predict(_train_features)
    actual_test_data = list(_test_target)
    predict_test_data = logistic_classifier.predict(_test_features)

    train_report = classification_report(actual_train_data, predict_train_data)
    test_report = classification_report(actual_test_data, predict_test_data)

    print('train_report')
    print(train_report)
    print('test_report')
    print(test_report)


def classify_by_bagging(_train_features, _train_target, _test_features, _test_target, _base_estimator, _n_estimators):
    bagging_classifier = BaggingClassifier(base_estimator=_base_estimator, n_estimators=_n_estimators,
                                           max_samples=0.5, max_features=0.5). \
        fit(_train_features, _train_target)

    actual_train_data = list(_train_target)
    predict_train_data = bagging_classifier.predict(_train_features)
    actual_test_data = list(_test_target)
    predict_test_data = bagging_classifier.predict(_test_features)

    train_report = classification_report(actual_train_data, predict_train_data)
    test_report = classification_report(actual_test_data, predict_test_data)

    print('train_report')
    print(train_report)
    print('test_report')
    print(test_report)


def classify_by_random_forest(_train_features, _train_target, _test_features, _test_target):
    accuracy_train_values = []
    accuracy_test_values = []
    random_forest_classifier = RandomForestClassifier(n_estimators=10, max_depth=2, random_state=0).fit(_train_features, train_target)

    actual_train_data = list(_train_target)
    predict_train_data = random_forest_classifier.predict(_train_features)
    actual_test_data = list(_test_target)
    predict_test_data = random_forest_classifier.predict(_test_features)

    for i in range(1, 50):
        random_forest_classifier = RandomForestClassifier(n_estimators=10, max_depth=i, random_state=0).fit(
            _train_features, train_target)

        accuracy_train_values.append(
            sklearn.metrics.accuracy_score(actual_train_data, random_forest_classifier.predict(_train_features)))
        accuracy_test_values.append(
            sklearn.metrics.accuracy_score(actual_test_data, random_forest_classifier.predict(_test_features)))

    x = list(range(50))
    plt.plot(accuracy_train_values)
    plt.plot(accuracy_test_values)
    plt.legend(["Train", "Test"])
    plt.xlabel('max_depth')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Diagram for Random Forest Classifier')
    plt.show()

    train_report = classification_report(actual_train_data, predict_train_data)
    test_report = classification_report(actual_test_data, predict_test_data)

    print('train_report')
    print(train_report)
    print('test_report')
    print(test_report)


def classify_by_voting(_train_features, _train_target, _test_features, _test_target, _n_estimators):
    clf1 = KNeighborsClassifier(n_neighbors=80)
    clf2 = tree.DecisionTreeClassifier(max_depth=4)
    clf3 = LogisticRegression()
    voting_classifier = VotingClassifier(estimators=[('knn', clf1), ('dt', clf2), ('lg', clf3)])
    voting_classifier = voting_classifier.fit(_train_features, _train_target)

    actual_train_data = list(_train_target)
    predict_train_data = voting_classifier.predict(_train_features)
    actual_test_data = list(_test_target)
    predict_test_data = voting_classifier.predict(_test_features)

    train_report = classification_report(actual_train_data, predict_train_data)
    test_report = classification_report(actual_test_data, predict_test_data)

    print('train_report')
    print(train_report)
    print('test_report')
    print(test_report)



main_features, main_target = process_input_data('data.csv')
train_features, test_features, train_target, test_target = get_train_test_data(main_features, main_target)

### Phase_0
#get_information_gain(main_features, main_target)

### Phase_1
ros = imblearn.over_sampling.RandomOverSampler()
#train_features, train_target = ros.fit_resample(train_features, train_target)

classify_by_KNN(train_features, train_target, test_features, test_target)
#classify_by_tree(train_features, train_target, test_features, test_target)
#classify_by_logistic(train_features, train_target, test_features, test_target)

### Phase_2
#classify_by_bagging(train_features, train_target, test_features, test_target, tree.DecisionTreeClassifier(max_depth=4), 20)
classify_by_bagging(train_features, train_target, test_features, test_target, KNeighborsClassifier(n_neighbors=80), 800)
#classify_by_voting(train_features, train_target, test_features, test_target, 100)
#classify_by_random_forest(train_features, train_target, test_features, test_target)
