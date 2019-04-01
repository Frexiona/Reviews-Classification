#!/usr/bin/env python
# -*- coding: utf-8 -*-

import urllib.request
import bs4
import re
import pandas as pd
import numpy as np

# Crawlers
# Functions:
# 1. Get the formatted html file from the specific URL
# 2. Get all the categories and their URLs from the index page
# 3. Get all the information for the specific category including name, location, tags and it's URL
class Crawlers():

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



class DataFrameGenerator():
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

    # Get all the reviews data from specific 
    def getDataFrameofThirdPage(self, df_from_second_page):
        c = Crawlers()

        urls = df_from_second_page['URL']
        for each_url in urls:
            html =

class TextCleaner():

    def __init__(self):
        pass


    def cleanStr(self, text):
        try:
            # python UCS-4 build
            high_points = re.compile(u'[\U00010000-\U0010ffff]')
        except re.error:
            # python UCS-2 build
            high_points = re.compile(u'[\uD800-\uDBFF][\uDC00-\uDFFF]')

        resolve_value = high_points.sub(u'', text)
        return resolve_value


if __name__ == '__main__':
    crawler = Crawlers()
    print(crawler.getDataFramefromTheme('Restaurants'))