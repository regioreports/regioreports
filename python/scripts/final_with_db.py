# -*- coding: utf-8 -*-
import sqlite3
from sqlite3 import Error

def create_database(db_file):
	try:
		conn = sqlite3.connect(db_file)
		c = conn.cursor()
		c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='regions'")
		if bool(c.fetchone()):
			return conn
		print("Tables doesn't exist.... Creating them")
		c.execute('''CREATE TABLE regions (
						id INTEGER PRIMARY KEY,
						yandex_id INTEGER,
						name TEXT NOT NULL,
						avito_id TEXT NOT NULL
					 )''')
		c.execute('''CREATE TABLE yandex_news (
						id INTEGER PRIMARY KEY AUTOINCREMENT,
						title TEXT NOT NULL,
						time_first TIMESTAMP NOT NULL,
						category TEXT NOT NULL,
						yandex_id INTEGER NOT NULL,
						FOREIGN KEY (yandex_id) REFERENCES regions(yandex_id)
					 )''')
		c.execute('''CREATE TABLE sources (
						id INTEGER PRIMARY KEY AUTOINCREMENT,
						news_id INTEGER NOT NULL,
						source_time TIMESTAMP NOT NULL,
						snippet TEXT NOT NULL,
						url TEXT NOT NULL,
						source_name TEXT NOT NULL,
						source_title TEXT NOT NULL,
						FOREIGN KEY(news_id) REFERENCES yandex_news(id)
					)''')
		c.execute('''CREATE TABLE avito_ads (
						id INTEGER PRIMARY KEY AUTOINCREMENT,
						data TEXT, 
						price TEXT,
						url TEXT NOT NULL,
						ad_date TIMESTAMP NOT NULL,
						title TEXT NOT NULL,
						avito_id INTEGER NOT NULL,
						FOREIGN KEY (avito_id) REFERENCES regions(avito_id)
					)''')
		#CREATE TABLE ready_reports (id INTEGER PRIMARY KEY AUTOINCREMENT,region_name TEXT NOT NULL,date_created TEXT NOT NULL,status TEXT NOT NULL,full_path TEXT,);
		conn.commit()
		return conn
	except Error as e:
		print(e)

import os
import sys
import re

import requests
import time
from bs4 import BeautifulSoup


PREFIX_YANDEX = 'https://news.yandex.ru/yandsearch?geonews={0}&rpt=nnews2&grhow=clutop&p='
PREFIX_YANDEX_BASE = 'https://news.yandex.ru'

PREFIX_AVITO = 'https://www.avito.ru/{0}/uslugi?p='

import datetime
def parseYandexDate(yandex_date):
	date = datetime.datetime.now()
	cur_time_min = int(yandex_date[-2:])
	cur_time_hh = int(yandex_date[-5:-3])
	if 'вчера' in yandex_date:
		date -= datetime.timedelta(1)
	elif '.' in yandex_date:
		dd=int(yandex_date[0:2])
		mm=int(yandex_date[3:5])
		yy=int(yandex_date[6:8])
		date=date.replace(month=mm, day=dd, year=2000+yy)
	elif len(yandex_date) > 5:
		months = ['янв', 'фев', 'март', 'апрел', 'мая', 'июня', 'июля', 'август', 'сентяб', 'октяб', 'нояб', 'декаб']
		dd = int(yandex_date[0:2])
		for i in range(0, len(months)):
			if months[i] in yandex_date:
				mm = i
				break
		date=date.replace(month=mm+1, day=dd)
	date=date.replace(hour=cur_time_hh, minute=cur_time_min, second=0, microsecond=0)
	return date


def download_yandex_page(url, db_cursor, region_id, yandex_id, region_name):
	print('Attempt to get ' + url + ' ...')
	try:
		r = requests.get(url)
		if r.status_code != 200:
			print('Error: ' + str(r.status_code))
			return
	except Error as e:
		print(e)
		return
	html = r.text
	#soup = BeautifulSoup(open("html.html"), 'html.parser')
	soup = BeautifulSoup(html, 'html.parser')
	news_urls = []
	titles = []
	print(html)
	for title in soup.find_all(class_='story-item'):
		news_urls.append(re.sub('\.\.+', '.', title.h2.a['href']) + '&content=alldocs' + '\n')
		titles.append(re.sub('\.\.+', '.', title.h2.a.get_text()) + '\n')
	print('Successefully loaded ' + str(len(news_urls)) + ' news_urls')
	if len(news_urls) == 0: 
		time.sleep(2000)
	return
	source_counter = 0
	news_counter = 0
	for i in range(0, len(news_urls)):
		time.sleep(120)
		current_news_url = PREFIX_YANDEX_BASE + news_urls[i]
		#print ('current_news_url = ' + current_news_url)
		try:
			r = requests.get(current_news_url)
			if r.status_code != 200:
				print('Error: ' + str(r.status_code))
				return
		except Error as e:
			print(e)
			return
		soup = BeautifulSoup(r.text, 'html.parser')
		#soup = BeautifulSoup(open("html_list.html"), 'html.parser')
		source_names = []
		for title in soup.find_all(class_='doc__agency'):
			source_names.append(re.sub('\.\.+', '.', title.get_text()) + '\n')
#		print("source names len = ", len(source_names))
#		print(source_names)
		source_times = []
		for title in soup.find_all(class_='doc__time'):
			cur_time = parseYandexDate(re.sub('\.\.+', '.', title.get_text()))
			source_times.append(cur_time)
#		print("source times len = ", len(source_times))
#		print(source_times)
		source_titles = []
		source_urls = []
		for title in soup.find_all(class_='doc__head'):
			source_urls.append(re.sub('\.\.+', '.', title.a['href']) + '\n')
			source_titles.append(re.sub('\.\.+', '.', title.h2.a.get_text()) + '\n')
#		print("source titles len = ", len(source_titles))
#		print(source_titles)
#		print("source urls len = ", len(source_urls))
#		print(source_urls)
		snippets = []     
		for title in soup.find_all(class_='doc__text'):
			snippets.append(re.sub('\.\.+', '.', title.get_text()) + '\n')
#		print("snippets len = ", len(snippets))
#		print(snippets)
		category = ""
		for title in soup.find_all(class_='link link_ajax link_theme_normal title__link i-bem'):
			category = re.sub('\.\.+', '.', title['title']) + '\n'
			break
#		print('category = ' + category)
		print('Successfully parsed all the information for the current news page!')
		print('Number of sources is: ' + str(len(source_urls)))
		
		does_news_exist = False

		news_id = ""
		for j in range(0, len(source_urls)):
			c.execute("SELECT news_id FROM sources WHERE url=(?)", (source_urls[j],))
			news_id = c.fetchone()
			if bool(news_id):
				does_news_exist = True
				break

		if does_news_exist:
			news_id = int(news_id[0])
			print("Oops, this news is " + str(news_id) + " already in the database...")

		if not does_news_exist:
			print("Inserting new news story and its sources...")
			if (len(source_times)) < 1:
				continue
			db_cursor.execute("INSERT INTO yandex_news (title, time_first, category, yandex_id) VALUES (?, ?, ?, ?)", 
													   (titles[i], source_times[-1], category, yandex_id))
			news_id = db_cursor.lastrowid
			news_counter += 1

		conn.commit()
		for j in range(0, len(source_urls)):
			c.execute("SELECT url FROM sources WHERE url=(?)", (source_urls[j],))
			vall = c.fetchone();
			if bool(vall):
				print("This source " + str(vall[0]) + " already exists in the database.....")
				continue
			if j < len(source_times) and j < len(snippets) and j < len(source_urls) and j < len(source_names) and j < len(source_titles):
				db_cursor.execute("INSERT INTO sources (news_id, source_time, snippet, url, source_name, source_title) VALUES (?, ?, ?, ?, ?, ?)", 
												   (news_id, source_times[j], snippets[j], source_urls[j], source_names[j], source_titles[j]))
			source_counter += 1
		conn.commit()
	print('Successfully updated the database for the region ' + region_name)
	print('Added ' + str(source_counter) + ' news sources and '+ str(news_counter) + ' news')
		
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
	#  вызываем main  c кодом региона, именем  и яндекс-ид
	'''
	import errno
	import fcntl
	try:
		ff = open ('lock', 'w')
		fcntl.flock (ff, fcntl.LOCK_EX | fcntl.LOCK_NB)
	except IOError as e:
		if e.errno == errno.EAGAIN:
			sys.stderr.write('[%s] Script already running.\n' % time.strftime ('%c'))
			sys.exit(-1)
	'''
	db_file = "database.db"
	region_ids = [26, 2, 23, 70]
	region_names = [u'Ставропольский край', u'Республика Башкортостан', u'Краснодарский край',  u'Томская область']
	yandex_ids = [36, 172, 35, 67]
	avito_ids = ['stavropolskiy_kray', 'bashkortostan', 'krasnodarskiy_kray', 'tomskaya_oblast']
	conn = create_database(db_file)
	c = conn.cursor()
	for i in range(0, len(region_ids)):
		c.execute("SELECT id FROM regions WHERE id=(?)", (region_ids[i],))
		if bool(c.fetchone()):
			print("Sorry, but the region " + region_names[i] + " already exists in the database..")
			continue
		c.execute("INSERT INTO regions VALUES (?, ?, ?, ?)", (region_ids[i], yandex_ids[i], region_names[i], avito_ids[i]))	
		print("inserted " + region_names[i])

	conn.commit()
	while True:
		for page in range(0, 30):
			for i in [0, 1, 2, 3]:
				download_avito_page(PREFIX_AVITO.format(avito_ids[i]) + str(page),
								c, region_ids[i], avito_ids[i], region_names[i])
				download_yandex_page(PREFIX_YANDEX.format(yandex_ids[i]) + str(page),
								c, region_ids[i], yandex_ids[i], region_names[i])
				time.sleep(120)
				conn.commit()
			time.sleep(120)
		print('Sleeping for 3600 sec until next loop of downloading information for regions...')
		time.sleep(3600)
	conn.close()
