import sqlite3
from sqlite3 import Error

def connect_to_db(db_file):
    try:
        conn = sqlite3.connect(db_file)
        c = conn.cursor()
        return conn
    except Error as e:
        print(e)

import os
import sys
import re

import requests
import time
from bs4 import BeautifulSoup

PREFIX_AVITO = 'https://www.avito.ru/{0}/uslugi?p='

import datetime

def download_avito_page(url, db_cursor, region_id, avito_id, region_name):
    print('Attempt to get ' + url)
    try:
        r = requests.get(url)
        if r.status_code != 200:
            print('Error: ' + str(r.status_code))
            return
    except Error as e:
        print(e)
        return
    html = r.text
    soup = BeautifulSoup(html, "html.parser")
    titles = []
    urls = []
    prices = []
    datas = []
    date = datetime.datetime.now()
    for title in soup.find_all(class_='description'):
        titles.append(" ".join(re.sub('\.\.+', '.', title.h3.get_text()).replace('\n','').split()) + '\n')
        urls.append(re.sub('\.\.+', '.', title.h3.a['href']) + '\n')
    for title in soup.find_all(class_='about'): #all is fine except the last one
        prices.append(" ".join(re.sub('\.\.+', '.', title.get_text()).replace('\n','').split()) + '\n')
    for title in soup.find_all(class_='data'): 
        datas.append(" ".join(re.sub('\.\.+', '.', title.get_text()).replace('\n','').split()) + '\n')

    for i in range(0, len(urls)):
        c.execute("SELECT url FROM avito_ads WHERE url=(?)", (urls[i],))
        vall = c.fetchone();
        if bool(vall):
            print("This source " + str(urls[i]) + " already exists in the database.....")
            continue
        db_cursor.execute("INSERT INTO avito_ads (data, price, url, ad_date, title, avito_id) VALUES (?, ?, ?, ?, ?, ?)", 
                                    (datas[i], prices[i], urls[i], date, titles[i], avito_id))
    print('Successfully added ' + str(len(urls)) + ' avito ads!!')

if __name__ == '__main__':
    db_file = "database.db"
    conn = connect_to_db(db_file)
    c = conn.cursor()
    region_ids = [26, 2, 23, 70]
    region_names = [u'Ставропольский край', u'Республика Башкортостан', u'Краснодарский край',  u'Томская область']
    yandex_ids = [36, 172, 35, 67]
    avito_ids = ['stavropolskiy_kray', 'bashkortostan', 'krasnodarskiy_kray', 'tomskaya_oblast']
    for i in [1,2,3,0]:
            for page in range(0, 20):
                download_avito_page(PREFIX_AVITO.format(avito_ids[i]) + str(page),
                                c, region_ids[i], avito_ids[i], region_names[i])
                conn.commit()
                time.sleep(60)
    conn.close()
