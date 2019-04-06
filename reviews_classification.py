#!/usr/bin/env python
# -*- coding: utf-8 -*-

import urllib.request
import bs4
import nltk
import re
import pandas as pd
import numpy as np

from nltk.corpus import stopwords


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
                comments.append(TextCleaner().cleanText(each_comment.get_text(strip=True)))

            for i in range(len(dates)):
                names.append(df_from_second_page['Name'][count])

            df_temp = pd.DataFrame(data=[name for name in names], columns=['Name'])
            df_temp['Commentator'] = np.array([commentator for commentator in commentators])
            df_temp['Date'] = np.array([date for date in dates])
            df_temp['Star'] = np.array([review_star for review_star in review_stars])
            df_temp['Comment'] = np.array([comment for comment in comments])

            df = df.append(df_temp, ignore_index=True)

        return df


# Text Cleaner
# Functions:
# 1.Clean the text
class TextCleaner:
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
    def preProcess(self):
        pass

    # Tokenlize the text
    # Return Format: list()
    def tokenlizeText(self, texts):
        items_list = list()
        for j in range(len(texts)):
            items_list.append(nltk.word_tokenize(texts[j]))

        return items_list

    # Remove the Punctuations, Stop Words and lowercase the item from the Item list
    # Return Format: list()
    def removePuncStopwords(self, items_list):
        for z in range(len(items_list)):
            items_list[z] = [w.lower() for w in items_list[z] if w.isalnum()]
            items_list[z] = [i for i in items_list[z] if (i not in stopwords.words('english'))]

        return items_list



if __name__ == '__main__':
    crawler = Crawlers()
    df_generator = DataFrameGenerator()

    # Get the Dataframes for the Hotels and Restaurants
    # df_hotels = crawler.getDataFramefromTheme('Hotels')
    # df_restaurants = crawler.getDataFramefromTheme('Restaurants')
    # df_hotels = df_generator.getDataFrameofThirdPage(df_hotels)
    # df_restaurants = df_generator.getDataFrameofThirdPage(df_restaurants)
    #
    # print(df_hotels)
    # print(df_restaurants)
    # df_hotels.to_csv('hotels.csv', sep=',', encoding='utf-8', index=False)
    # df_restaurants.to_csv('restaurants.csv', sep=',', encoding='utf-8', index=False)
    df_hotels = pd.read_csv('hotels.csv', sep=',')
    df_restaurants = pd.read_csv('restaurants.csv', sep=',')

    print(df_hotels)
    print(df_restaurants)