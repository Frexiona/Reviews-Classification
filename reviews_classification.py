#!/usr/bin/env python
# -*- coding: utf-8 -*-

import urllib.request
import bs4
import nltk
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Ignore the future warnings raised by Pandas
warnings.simplefilter(action='ignore', category=FutureWarning)

# Crawlers
# Functions:
# 1. Get the formatted html file from the specific URL
# 2. Get all the categories and their URLs from the index page
# 3. Get all the information for the specific category including name, location, tags and it's URL
class Crawlers:

    # Initialize the necessary data
    def __init__(self):
        self.key = "User-Agent"
        self.value = "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.106 Safari/537.36"
        self.index_url = 'http://mlg.ucd.ie/modules/yalp/'

    # Get the formatted html file from the specific URL
    # Return Format: BeautifulSoup Object
    def urlOpen(self, url):
        req = urllib.request.Request(url)
        response = urllib.request.urlopen(req)
        html = response.read().decode('utf-8')
        html = bs4.BeautifulSoup(html, 'html5lib')

        return html

    # Get all the categories and their URLs from the index page
    # Return Format: DataFrame
    def getUrlfromIndex(self):
        urls, titles, reviews = list(), list(), list()

        html = self.urlOpen(self.index_url)
        for index in html.findAll('a', href=True):
            urls.append(self.index_url + index['href'])
        for text in html.findAll('div', {'class': 'cat'}):
            text = text.get_text(strip=True).replace('Category: ', '').split('(')
            titles.append(text[0])
            reviews.append(text[1].replace(' reviews)', ''))

        df = pd.DataFrame(data=[title for title in titles], columns=['Theme'])
        df['URL'] = np.array([url for url in urls])
        df['Reviews Count'] = np.array([review for review in reviews])

        return df

    # Get all the information for the specific category including name, location, tags and it's URL
    # Return Format: DataFrame
    def getDataFramefromTheme(self, theme_name):
        dfg = DataFrameGenerator()

        if theme_name == 'Automotive':
            return dfg.getDataFrameofSecondPage(0)

        elif theme_name == 'Bars':
            return dfg.getDataFrameofSecondPage(1)

        elif theme_name == 'Health':
            return dfg.getDataFrameofSecondPage(2)

        elif theme_name == 'Hotels':
            return dfg.getDataFrameofSecondPage(3)
        elif theme_name == 'Restaurants':
            return dfg.getDataFrameofSecondPage(4)

# Data Frame Generator
# Functions:
# 1. Format the information for the specific category including name, location, tags and it's URL into DataFrame
# 2. Get all the data including names, commentators, dates, stars and comments from specific category
class DataFrameGenerator:
    def __init__(self):
        pass

    # Format the information for the specific category including name, location, tags and it's URL into DataFrame
    # Return Format: DataFrame
    def getDataFrameofSecondPage(self, url_number):
        c = Crawlers()

        df = c.getUrlfromIndex()
        url = df['URL'][url_number]
        html = c.urlOpen(url)

        titles, urls, locations, tags = list(), list(), list(), list()

        for index in html.findAll('a', href=True):
            urls.append(c.index_url + index['href'])

        for title in html.find_all('a'):
            titles.append(title.string)

        for location in html.find_all():
            pass

        for tags in html.find_all():
            pass

        df = pd.DataFrame(data=[title for title in titles], columns=['Name'])
        # df['Location'] = np.array([location for location in locations])
        # df['tag'] = np.array([tag for tag in tags])
        df['URL'] = np.array([url for url in urls])

        return df

    # Get all the data including names, commentators, dates, stars and comments from specific category
    # Return Format: DataFrame
    def getDataFrameofThirdPage(self, df_from_second_page):
        c = Crawlers()

        urls = df_from_second_page['URL']
        df = pd.DataFrame(columns=['Name', 'Commentator', 'Date', 'Star', 'Comment'])
        for count in range(len(urls)):
            html = c.urlOpen(urls[count])
            names, dates, commentators, review_stars, comments = list(), list(), list(), list(), list()

            for each_date_commentator in html.findAll('p', {'class': 'review-top'}):
                each_date_commentator = each_date_commentator.get_text(strip=True).split('by')
                dates.append(each_date_commentator[0].replace('Reviewed on ', ''))
                commentators.append(each_date_commentator[1])

            for each_review_star in html.findAll('p', {'class': 'stars'}):
                each_review_star = each_review_star.find('img')['alt']
                if each_review_star == '5-star' or each_review_star == '4-star':
                    each_review_star = 'positive'
                else:
                    each_review_star = 'negative'
                review_stars.append(each_review_star)

            for each_comment in html.findAll('p', {'class': 'text'}):
                comments.append(TextProcessor().cleanText(each_comment.get_text(strip=True)))

            for i in range(len(dates)):
                names.append(df_from_second_page['Name'][count])

            df_temp = pd.DataFrame(data=[name for name in names], columns=['Name'])
            df_temp['Commentator'] = np.array([commentator for commentator in commentators])
            df_temp['Date'] = np.array([date for date in dates])
            df_temp['Star'] = np.array([review_star for review_star in review_stars])
            df_temp['Comment'] = np.array([comment for comment in comments])

            df = df.append(df_temp, ignore_index=True)

        return df

    # Return the cleaned items with DataFrame format
    # Return Format: DataFrame
    def getDataFramefromList(self, items_list, df_lables):
        df = pd.DataFrame(data=[" ".join(review) for review in items_list], columns=['clean_comments'])
        df['lables'] = df_lables.replace('positive', 1).replace('negative', 0)

        return df


# Text Cleaner
# Functions:
# 1.Clean the text
class TextProcessor:
    def __init__(self):
        pass

    def cleanText(self, text):
        try:
            # python UCS-4 build
            high_points = re.compile(u'[\U00010000-\U0010ffff]')
        except re.error:
            # python UCS-2 build
            high_points = re.compile(u'[\uD800-\uDBFF][\uDC00-\uDFFF]')

        resolve_value = high_points.sub(u'', text)
        return resolve_value

    # Combine the preprocess functions into one function
    def preProcess(self, df):
        df = df.values
        items_list = list()

        tokenlizer = RegexpTokenizer(r'\w+')
        lemmatizer = WordNetLemmatizer()

        # Tokenlize the comments and remove the punctuations
        for j in range(len(df)):
            items_list.append(tokenlizer.tokenize(df[j]))

        # Remove the stop words and lemmatisation
        for z in range(len(items_list)):
            for index in range(len(items_list[z])):
                if items_list[z][index].isalnum():
                    # Lowercase and lemmatize the words
                    items_list[z][index] = lemmatizer.lemmatize(items_list[z][index].lower())

            items_list[z] = [i for i in items_list[z] if (i not in stopwords.words('english'))]

        return items_list

    def getTFIDF(self, df):
        vectorizer = TfidfVectorizer()
        df_with_tfidf = vectorizer.fit_transform(df.values)

        return df_with_tfidf.toarray()


    # Tokenlize the text
    # Return Format: list()
    def tokenlizeText(self, df):
        items_list = list()
        for j in range(len(df)):
            items_list.append(nltk.word_tokenize(df[j]))

        return items_list

    # Remove the Punctuations, Stop Words and lowercase the item from the Item list
    # Return Format: list()
    def removePuncStopwords(self, items_list):
        lemmatizer = WordNetLemmatizer()
        for z in range(len(items_list)):
            for index in range(len(items_list[z])):
                lemmatizer.lemmatize(items_list[z][index])
                if items_list[z][index].isalnum():
                    items_list[z][index].lower()

            items_list[z] = [i for i in items_list[z] if (i not in stopwords.words('english'))]

        return items_list


class DataProcess:
    def __init__(self):
        # 1. Knn
        self.knn_model = KNeighborsClassifier(n_neighbors=3)
        # 2. Decision Tree
        self.dt_model = DecisionTreeClassifier()
        # 3. Naive Bayes
        self.nb_model = MultinomialNB()
        # 4. SVM
        self.svm_model = SVC()
        # 5. Logistic Regression
        self.lr_model = linear_model.LogisticRegression()


    def getTrainedModel(self, train_data, test_data):
        knn_trained_model = self.knn_model.fit(train_data, test_data)
        dt_trained_model = self.dt_model.fit(train_data, test_data)
        nb_trained_model = self.nb_model.fit(train_data, test_data)
        svm_trained_model = self.svm_model.fit(train_data, test_data)
        lr_trained_model = self.lr_model.fit(train_data, test_data)

        return knn_trained_model, dt_trained_model, nb_trained_model, svm_trained_model, lr_trained_model

    def getPredictScoreandCM(self, train_data, test_data, test_size):
        # Split data for generalization test, hotel_data_test and hotel_data_label_test are the unused data for the model
        training_data, test_training_data, target_data, test_target_data = train_test_split(train_data, test_data, test_size=test_size)

        knn_trained_model, dt_trained_model, nb_trained_model, svm_trained_model, lr_trained_model \
            = self.getTrainedModel(training_data, target_data)

        # Accuracy for models on unseen data(Data Scope: 20%)
        knn_score = accuracy_score(test_target_data, knn_trained_model.predict(test_training_data))
        dt_score = accuracy_score(test_target_data, dt_trained_model.predict(test_training_data))
        nb_score = accuracy_score(test_target_data, nb_trained_model.predict(test_training_data))
        svm_score = accuracy_score(test_target_data, svm_trained_model.predict(test_training_data))
        lr_score = accuracy_score(test_target_data, lr_trained_model.predict(test_training_data))

        return knn_score, dt_score, nb_score, svm_score, lr_score

    def getConfusionMatrix(self, train_data, test_data, test_size):
        # Split data for generalization test, hotel_data_test and hotel_data_label_test are the unused data for the model
        training_data, test_training_data, target_data, test_target_data = train_test_split(train_data, test_data, test_size=test_size)

        knn_trained_model, dt_trained_model, nb_trained_model, svm_trained_model, lr_trained_model \
            = self.getTrainedModel(training_data, target_data)

        knn_cm = classification_report(test_target_data, knn_trained_model.predict(test_training_data))
        dt_cm = confusion_matrix(test_target_data, dt_trained_model.predict(test_training_data), labels=[1, -1])
        nb_cm = confusion_matrix(test_target_data, nb_trained_model.predict(test_training_data), labels=[1, -1])
        svm_cm = confusion_matrix(test_target_data, svm_trained_model.predict(test_training_data), labels=[1, -1])
        lr_cm = confusion_matrix(test_target_data, lr_trained_model.predict(test_training_data), labels=[1, -1])

        return knn_cm, dt_cm, nb_cm, svm_cm, lr_cm

    def getPredictScoreCV(self, train_data, test_data):
        # Cross Validation on different models (Data Scope: All) (Cross Validation: 10)
        knn_cv_score = cross_val_score(self.knn_model, train_data, test_data, cv=10, scoring="accuracy").mean()
        dt_cv_score = cross_val_score(self.dt_model, train_data, test_data, cv=10, scoring="accuracy").mean()
        nb_cv_score = cross_val_score(self.nb_model, train_data, test_data, cv=10, scoring="accuracy").mean()
        svm_cv_score = cross_val_score(self.svm_model, train_data, test_data, cv=10, scoring="accuracy").mean()
        lr_cv_score = cross_val_score(self.lr_model, train_data, test_data, cv=10, scoring="accuracy").mean()

        return knn_cv_score, dt_cv_score, nb_cv_score, svm_cv_score, lr_cv_score

    def getPredictScoreonB(self, train_data_A, test_data_A, train_data_B, test_data_B):
        knn_trained_model, dt_trained_model, nb_trained_model, svm_trained_model, lr_trained_model \
            = self.getTrainedModel(train_data_A, test_data_A)

        knn_score = accuracy_score(test_data_B, knn_trained_model.predict(train_data_B))
        dt_score = accuracy_score(test_data_B, dt_trained_model.predict(train_data_B))
        nb_score = accuracy_score(test_data_B, nb_trained_model.predict(train_data_B))
        svm_score = accuracy_score(test_data_B, svm_trained_model.predict(train_data_B))
        lr_score = accuracy_score(test_data_B, lr_trained_model.predict(train_data_B))

        return knn_score, dt_score, nb_score, svm_score, lr_score


if __name__ == '__main__':
    crawler = Crawlers()
    df_generator = DataFrameGenerator()
    text_cleaner = TextProcessor()
    data_process = DataProcess()

    # Get the Dataframes for the Hotels and Restaurants
    df_hotels = crawler.getDataFramefromTheme('Hotels')
    df_restaurants = crawler.getDataFramefromTheme('Restaurants')
    df_hotels = df_generator.getDataFrameofThirdPage(df_hotels)
    df_restaurants = df_generator.getDataFrameofThirdPage(df_restaurants)
    # df_hotels.to_csv('hotels.csv', sep=',', encoding='utf-8', index=False)
    # df_restaurants.to_csv('restaurants.csv', sep=',', encoding='utf-8', index=False)
    # df_hotels = pd.read_csv('hotels.csv', sep=',')
    # df_restaurants = pd.read_csv('restaurants.csv', sep=',')

    # Hotel Data Process Part
    review_document_hotels_data = text_cleaner.preProcess(df_hotels['Comment'])
    df_hotel_cleancomments = df_generator.getDataFramefromList(review_document_hotels_data, df_hotels['Star'])

    # Restaurant Data Process Part
    review_document_restaurants_data = text_cleaner.preProcess(df_restaurants['Comment'])
    df_restaurants_cleancomments = df_generator.getDataFramefromList(review_document_restaurants_data, df_restaurants['Star'])

    # Get the number of the rows of the data
    # Split the data
    data_split = len(df_hotel_cleancomments)
    all_data_cleancomments = df_hotel_cleancomments.append(df_restaurants_cleancomments, ignore_index=True)
    all_data = text_cleaner.getTFIDF(all_data_cleancomments['clean_comments'])
    all_data_label = all_data_cleancomments['lables'].to_numpy()
    hotel_data = all_data[:data_split, :]
    hotel_data_label = all_data_label[:data_split]
    restaurants_data = all_data[data_split:, :]
    restaurants_data_label = all_data_label[data_split:]

    # Accuracy for models on unseen data, Hotel(Data Scope: 20%)
    knn_hotel_score, dt_hotel_score, nb_hotel_score, svm_hotel_score, lr_hotel_score = \
        data_process.getPredictScore(hotel_data, hotel_data_label, 0.2)

    # Confusion Matrix for models on unseen data, Hotel(Data Scope: 20%)
    knn_cm, dt_cm, nb_cm, svm_cm, lr_cm = data_process.getConfusionMatrix(hotel_data, hotel_data_label, 0.2)
    print("Confusion Matrix for Logistic Regression")
    print(lr_cm)

    # Cross Validation on different models, Hotel(Data Scope: All) (Cross Validation: 10)
    knn_hotel_cv_score, dt_hotel_cv_score, nb_hotel_cv_score, svm_hotel_cv_score, lr_hotel_cv_score = \
        data_process.getPredictScoreCV(hotel_data, hotel_data_label)

    # Data visualization of hotel data
    # Graph 1: Accuracy for models on unseen data (Hotel)
    model_names = ['KNN', 'Decision Tree', 'Naive Bayes', 'SVM', 'Logistic Regression']
    y_pos = np.arange(len(model_names))
    hotel_model_score = [knn_hotel_score, dt_hotel_score, nb_hotel_score, svm_hotel_score, lr_hotel_score]
    plt.bar(y_pos, hotel_model_score, align='center', alpha=0.5)
    plt.xticks(y_pos, model_names, fontsize=10, rotation=60)
    plt.xlabel('Models', fontsize=10)
    plt.ylabel('Accuracy Score', fontsize=10)
    plt.title('Accuracy for different models on unseen data (Hotel)')
    for a, b in enumerate(hotel_model_score):
        plt.text(a, b, '%.03f' % b, ha='center')
    plt.show()

    # Graph 2: Accuracy for models by using cross validation (Hotel)
    hotel_model_cv_score = [knn_hotel_cv_score, dt_hotel_cv_score, nb_hotel_cv_score, svm_hotel_cv_score, lr_hotel_cv_score]
    plt.bar(y_pos, hotel_model_cv_score, align='center', alpha=0.5)
    plt.xticks(y_pos, model_names, fontsize=10, rotation=60)
    plt.xlabel('Models', fontsize=10)
    plt.ylabel('Accuracy Score (Cross Validation)', fontsize=10)
    plt.title('Accuracy for different models (Cross Validation, Hotel)')
    for a, b in enumerate(hotel_model_cv_score):
        plt.text(a, b, '%.03f' % b, ha='center')
    plt.show()

    # Accuracy for models on unseen data, Restaurant(Data Scope: 20%)
    knn_restaurants_score, dt_restaurants_score, nb_restaurants_score, svm_restaurants_score, lr_restaurants_score = \
        data_process.getPredictScore(restaurants_data, restaurants_data_label, 0.2)

    # Confusion Matrix for models on unseen data, Hotel(Data Scope: 20%)
    knn_cm, dt_cm, nb_cm, svm_cm, lr_cm = data_process.getConfusionMatrix(hotel_data, hotel_data_label, 0.2)
    print(knn_cm)

    # Cross Validation on different models, Restaurant(Data Scope: All) (Cross Validation: 10)
    knn_restaurants_cv_score, dt_restaurants_cv_score, nb_restaurants_cv_score, svm_restaurants_cv_score, lr_restaurants_cv_score = \
        data_process.getPredictScoreCV(restaurants_data, restaurants_data_label)

    # Graph 3: Accuracy for models on unseen data (Restaurant)
    restaurants_model_score = [knn_restaurants_score, dt_restaurants_score, nb_restaurants_score, svm_restaurants_score, lr_restaurants_score]
    plt.bar(y_pos, restaurants_model_score, align='center', alpha=0.5)
    plt.xticks(y_pos, model_names, fontsize=10, rotation=60)
    plt.xlabel('Models', fontsize=10)
    plt.ylabel('Accuracy Score', fontsize=10)
    plt.title('Accuracy for different models on unseen data (Restaurant)')
    for a, b in enumerate(restaurants_model_score):
        plt.text(a, b, '%.03f' % b, ha='center')
    plt.show()

    # Graph 4: Accuracy for models by using cross validation (Restaurant)
    restaurants_model_cv_score = [knn_restaurants_cv_score, dt_restaurants_cv_score, nb_restaurants_cv_score, svm_restaurants_cv_score, lr_restaurants_cv_score]
    plt.bar(y_pos, restaurants_model_cv_score, align='center', alpha=0.5)
    plt.xticks(y_pos, model_names, fontsize=10, rotation=60)
    plt.xlabel('Models', fontsize=10)
    plt.ylabel('Accuracy Score (Cross Validation)', fontsize=10)
    plt.title('Accuracy for different models (Cross Validation, Restaurant)')
    for a, b in enumerate(restaurants_model_cv_score):
        plt.text(a, b, '%.03f' % b, ha='center')
    plt.show()

    # Run restaurant data on hotel model
    # Train the model by all the hotel data
    knn_hotel_score_onRe, dt_hotel_score_onRe, nb_hotel_score_onRe, svm_hotel_score_onRe, lr_hotel_score_onRe = \
        data_process.getPredictScoreonB(hotel_data, hotel_data_label, restaurants_data, restaurants_data_label)

    # Graph 5: Accuracy for different hotel models on restaurant data
    hotel_model_score_onRe = [knn_hotel_score_onRe, dt_hotel_score_onRe, nb_hotel_score_onRe, svm_hotel_score_onRe, lr_hotel_score_onRe]
    plt.bar(y_pos, hotel_model_score_onRe, align='center', alpha=0.5)
    plt.xticks(y_pos, model_names, fontsize=10, rotation=60)
    plt.xlabel('Models', fontsize=10)
    plt.ylabel('Accuracy Score (hotel model on restaurant data)', fontsize=10)
    plt.title('Accuracy for different hotel models on restaurant data')
    for a, b in enumerate(hotel_model_score_onRe):
        plt.text(a, b, '%.03f' % b, ha='center')
    plt.show()

    # Run hotel data on restaurant model
    knn_hotel_score_onHo, dt_hotel_score_onHo, nb_hotel_score_onHo, svm_hotel_score_onHo, lr_hotel_score_onHo = \
        data_process.getPredictScoreonB(restaurants_data, restaurants_data_label, hotel_data, hotel_data_label)

    # Graph 6: Accuracy for different restaurant models on hotel data
    hotel_model_score_onHo = [knn_hotel_score_onHo, dt_hotel_score_onHo, nb_hotel_score_onHo, svm_hotel_score_onHo,
                              lr_hotel_score_onHo]
    plt.bar(y_pos, hotel_model_score_onHo, align='center', alpha=0.5)
    plt.xticks(y_pos, model_names, fontsize=10, rotation=60)
    plt.xlabel('Models', fontsize=10)
    plt.ylabel('Accuracy Score (restaurant model on hotel data)', fontsize=10)
    plt.title('Accuracy for different restaurant models on hotel data')
    for a, b in enumerate(hotel_model_score_onHo):
        plt.text(a, b, '%.03f' % b, ha='center')
    plt.show()