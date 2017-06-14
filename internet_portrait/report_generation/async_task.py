# -*- coding: utf-8 -*-
import os, django,sys
sys.path.append('/Users/ilonapapava/Programming/papava.me/internet_portrait')
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "internet_portrait.settings")
django.setup()

from django.http import HttpResponse
from django.template import Context
from django.template.loader import get_template
from subprocess import Popen, PIPE, CalledProcessError
import tempfile

import sqlite3
import pymorphy2
from report_generation.models import Regions, YandexNews, Sources, AvitoAds
from django.db.models import Count

from igraph import *
import collections
from collections import Counter
import numpy as np

import time
import matplotlib.pyplot as plt
import re

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

from datetime import date, timedelta

def generate_report(region_name):
	context = {'content': region_name}
	all_news = YandexNews.objects.filter(yandex__name=context['content'])
	context['number_of_news'] = len(all_news)
	all_sources = Sources.objects.filter(news__yandex__name=context['content'])
	context['number_of_sources'] = len(all_sources)
	sorted_objects = YandexNews.objects.filter(yandex__name=context['content']).order_by('time_first')
	from babel.dates import format_datetime
	context['period_from'] = format_datetime(sorted_objects[0].time_first, locale='ru_RU.UTF-8')
	context['period_to'] = format_datetime(sorted_objects[len(sorted_objects) - 1].time_first, locale='ru_RU.UTF-8')
	context['date_report_generated'] = time.strftime("%x")
	top_news = YandexNews.objects.filter(yandex__name=context['content']) \
					.extra(
						select={
							'num_sources': 'SELECT COUNT(*) FROM sources WHERE sources.news_id = yandex_news.id',
						},
						order_by=['-num_sources']
					)
	NUM_OF_NEWS = 10
	print('Creating top news stats images...')
	list_of_images = []
	dicts_for_news = []

	for i in range(NUM_OF_NEWS):
		yan_id = top_news[i].id

		num_per_date = collections.defaultdict(dict)
		delta = sorted_objects[len(sorted_objects) - 1].time_first.date() - sorted_objects[0].time_first.date()
		for j in range(delta.days + 1):
			cur_date = sorted_objects[0].time_first.date() + timedelta(days=j)
			num_per_date[cur_date] = 0 

		cur_s = Sources.objects.filter(news_id=yan_id).order_by('source_time')
		for el in cur_s:
			num_per_date[el.source_time.date()] += 1
		dicts_for_news.append(dict(num_per_date))

	print('Making individual images...')
	sources_counter = 0
	for i in range(NUM_OF_NEWS):
		D = dicts_for_news[i]
		plt.plot(list(D.keys()), list(D.values()), label = u'Новость #' + str(i + 1))
		plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
		plt.ylabel(u'Количество источников')
		plt.xlabel(u'Дата')
		plt.title('Общее количество источников: ' + str(top_news[i].num_sources))
		sources_counter += top_news[i].num_sources
		plt.savefig('/Users/ilonapapava/Programming/papava.me/internet_portrait/static/pics/img'+str(top_news[i].id)+'.png', bbox_inches='tight')
		list_of_images.append('img' + str(top_news[i].id)+'.png')
		plt.clf()

	print('Making full image...')
	for i in range(NUM_OF_NEWS):
		D = dicts_for_news[i]
		plt.plot(list(D.keys()), list(D.values()), label = u'Новость #' + str(i + 1))
		plt.ylabel(u'Количество источников')
		plt.xlabel(u'Дата')
		plt.title('Общее количество источников: ' + str(sources_counter))

	plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
	plt.savefig('/Users/ilonapapava/Programming/papava.me/internet_portrait/static/pics/all_news.png', bbox_inches='tight')	
	context['top_news'] = zip([el.title for el in top_news[:NUM_OF_NEWS]], [el.num_sources for el in top_news[:NUM_OF_NEWS]], list_of_images)

	all_source_titles = ""
	all_sources = all_sources.order_by('source_time')
	for source in all_sources:
	 	all_source_titles += source.source_title
	all_source_titles_normalized = normalizeText(all_source_titles, True, True)
	print('Working on wordcloud...')
	wordcloud_words = wordCloudBuilder(all_source_titles_normalized, '/Users/ilonapapava/Programming/papava.me/internet_portrait/static/pics/wordcloud.png')
	context['wordcloud_words'] = zip(list(wordcloud_words.keys())[5:15], list(wordcloud_words.values())[5:15])
	print('Success with wordcloud!')
	print('Building graph...')
	graphBuilder(all_source_titles_normalized)
	print('Success with graph!')

	#######movingAverage(all_sources, all_source_titles_normalized)

	print('Avito time...')
	avito_objects = AvitoAds.objects.filter(avito__name=context['content'])

	all_avito_titles=''
	for i in avito_objects:
		all_avito_titles += i.title	

	vectorizer = TfidfVectorizer(max_features=10000, max_df=0.5, min_df=2)
	

	f = open('/Users/ilonapapava/Programming/papava.me/internet_portrait/static/training_set/training_set.txt', 'r')
	training_set = f.readlines()

	training_set_result = [int(x.split(' ')[-1]) for x in training_set]
	training_set = [' '.join(x.split(' ')[:-1]) for x in training_set]

	#x_train = vectorizer.fit_transform(training_set)
	y_train = training_set_result
	dataset = normalizeText(all_avito_titles, False, False, True).split('\n')
	#x_test = vectorizer.fit_transform(dataset)

	full_set = training_set + dataset
	full_set_transformed = vectorizer.fit_transform(full_set)

	x_train = full_set_transformed[:len(training_set)]
	x_test = full_set_transformed[len(training_set):]

#	clf_logreg = LogisticRegression()
#	clf_logreg.fit(x_train, y_train)
#	y_test = clf_logreg.predict(x_test)

	clf_rf = RandomForestClassifier(n_estimators = 750)
	clf_rf.fit(x_train, y_train)
	y_test_rf = clf_rf.predict(x_test)

	NUMBER_OF_TOPICS = 13

	for cur in range(NUMBER_OF_TOPICS):
		print(cur, len(np.where(y_test_rf == cur)[0]))
	print('Success with topics prediction!!')
	topic_to_counter = {
		'Автомобили, трансфер, переезд': len(np.where(y_test_rf == 0)[0]),
		'Ремонт, строительство': len(np.where(y_test_rf == 1)[0]),
		'Уборка': len(np.where(y_test_rf == 2)[0]),
		'Красота и здоровье': len(np.where(y_test_rf == 3)[0]),
		'Сад, огород': len(np.where(y_test_rf == 4)[0]),
		'Бытовая и компьютерная техника': len(np.where(y_test_rf == 5)[0]),
		'Юриспруденция, услуги адвоката': len(np.where(y_test_rf == 6)[0]),
		'Репетиторство': len(np.where(y_test_rf == 7)[0]),
		'Выпечка, организация праздников': len(np.where(y_test_rf == 8)[0]),
		'Свадьбы, свадебные фотографии': len(np.where(y_test_rf == 9)[0]),
		'Вывоз мусора': len(np.where(y_test_rf == 11)[0]),
		'Мебель, реставрация мебели': len(np.where(y_test_rf == 12)[0]),
		'Прочее': len(np.where(y_test_rf == 10)[0])
	}
	print('Picturing wordclouds for avito topics...')
	wordclouds_path = []
	for cur in range(NUMBER_OF_TOPICS):
		current_data = ' '.join(dataset[x] for x in np.where(y_test_rf == cur)[0])
		current_path = '/Users/ilonapapava/Programming/papava.me/internet_portrait/static/pics/wordcloud_avito' + str(cur) + '.png'	
		wordCloudBuilder(current_data, current_path)
		wordclouds_path.append(current_path)

	context['avito_data'] = zip(list(topic_to_counter.keys()), list(topic_to_counter.values()), wordclouds_path)
	context['avito_total'] = len(dataset)
	print('Success with avito wordclouds!')
	#'/Users/ilonapapava/Programming/papava.me/internet_portrait/static/pics/wordcloud.png'
	context['map_path'] = Regions.objects.filter(name=context['content'])[0].avito_id + '.png'

# 0 -- 0,1,5,19,26,42 — трансфер, авто, переезд
# 1 -- 2,4,7,8,10,15,17,18,20,21,23,25,32,34,35,40,48,49— ремонт, строительство
# 2 -- 3,24 — уборка,
# 3 -- 6,12,33,46 — бьюти
# 4 -- 11,22,28,44 — сад, огород
# 5 -- 13,16,29,41,43— бытовая, компьютерная техника
# 6 -- 14,39 — юриспруденция, адвокат
# 7 -- 27,37 — услуги репетиторства
# 8 -- 30,36 — торты, конфета-букет
# 9 -- 45 — фотограф свадебный
# 10 -- 9 — прочее(удалить?)
# 11 -- 31 — вывоз мусора
# 12 -- 38,47 — реставрация мебели, мебель перетяжка

	print("Rendering template...")
	template = get_template('my_latex_template.tex')
	rendered_tpl = template.render(context).encode('utf-8')
	BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
	cur_time = str(time.time())
	with tempfile.TemporaryDirectory() as tempdir:  
		for ppp in range(2):
			try:
				process = Popen(
					['pdflatex', '-output-directory', tempdir],
					stdin=PIPE,
					stdout=PIPE,
					stderr=PIPE,
					cwd=r'/Users/ilonapapava/Programming/papava.me/internet_portrait/static/pics'
				)
				process.communicate(rendered_tpl)
			except:
				print("FUCK THAT SHIT")
		os.rename(os.path.join(tempdir, 'texput.pdf'), '/Users/ilonapapava/Programming/papava.me/internet_portrait/static/ready_reports/textput' + cur_time + '.pdf')
	return 'static/ready_reports/textput' + cur_time + '.pdf'

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

stop_words=['изз','под','без', 'а', 'будем', 'будет', 'будете', 'будешь', 'буду', 'будут', 'будучи', 'будь', 'будьте', 'бы', 'был', 'была', 'были', 'было', 'быть', 'в', 'вам', 'вами', 'вас', 'весь', 'во', 'вот', 'все', 'всё', 'всего', 'всей', 'всем', 'всём', 'всеми', 'всему', 'всех', 'всею', 'всея', 'всю', 'вся', 'вы', 'да', 'для', 'до', 'его', 'едим', 'едят', 'ее', 'её', 'ей', 'ел', 'ела', 'ем', 'ему', 'емъ', 'если', 'ест', 'есть', 'ешь', 'еще', 'ещё', 'ею', 'же', 'за', 'и', 'из', 'или', 'им', 'ими', 'имъ', 'их', 'к', 'как', 'кем', 'ко', 'когда', 'кого', 'ком', 'кому', 'комья', 'которая', 'которого', 'которое', 'которой', 'котором', 'которому', 'которою', 'которую', 'которые', 'который', 'которым', 'которыми', 'которых', 'кто', 'меня', 'мне', 'мной', 'мною', 'мог', 'моги', 'могите', 'могла', 'могли', 'могло', 'могу', 'могут', 'мое', 'моё', 'моего', 'моей', 'моем', 'моём', 'моему', 'моею', 'можем', 'можно', 'может', 'можете', 'можешь', 'мои', 'мой', 'моим', 'моими', 'моих', 'мочь', 'мою', 'моя', 'мы', 'на', 'нам', 'нами', 'нас', 'наса', 'наш', 'наша', 'наше', 'нашего', 'нашей', 'нашем', 'нашему', 'нашею', 'наши', 'нашим', 'нашими', 'наших', 'нашу', 'не', 'него', 'нее', 'неё', 'ней', 'нем', 'нём', 'нему', 'нет', 'нею', 'ним', 'ними', 'них', 'но', 'о', 'об', 'один', 'одна', 'одни', 'одним', 'одними', 'одних', 'одно', 'одного', 'одной', 'одном', 'одному', 'одною', 'одну', 'он', 'она', 'оне', 'они', 'оно', 'от', 'по', 'при', 'с', 'сам', 'сама', 'сами', 'самим', 'самими', 'впрочем', 'самих', 'само', 'самого', 'самом', 'самому', 'саму', 'свое', 'своё', 'своего', 'своей', 'своем', 'своём', 'своему', 'своею', 'свои', 'свой', 'своим', 'своими', 'своих', 'свою', 'своя', 'себе', 'себя', 'собой', 'собою', 'та', 'так', 'такая', 'такие', 'таким', 'такими', 'таких', 'такого', 'такое', 'такой', 'таком', 'такому', 'такою', 'такую', 'те', 'тебе', 'тебя', 'тем', 'теми', 'тех', 'то', 'тобой', 'тобою', 'того', 'той', 'только', 'том', 'томах', 'тому', 'тот', 'тою', 'ту', 'ты', 'у', 'уже', 'чего', 'чем', 'чём', 'чему', 'что', 'чтобы', 'эта', 'эти', 'этим', 'этими', 'этих', 'это', 'этого', 'этой', 'этом', 'этому', 'этот', 'этою', 'эту', 'я', 'один', 'два', 'три', 'четыре', 'пять', 'шесть', 'семь', 'восемь', 'девять', 'десять', 'трое']
stop_words_regions = ['человек','республика','башкирский','апрель','май','июнь','уфимец','стерлитамак','уфимский','житель','женщина','мужчина','геленджик','город','оренбург','томич','область','томский','томск','ставрополье', 'пятигорск', 'ставрополь', 'кисловодск', 'ставрополец', 'невинномысск', 'ессентуки',  'ставропольский', 'край', 'краснодар', 'сочи', 'кубань', 'краснодарский', 'новороссийск', 'башкортостан', 'башкирия', 'уфа']
stop_words_avito = ['человек','республика','башкирский','апрель','май','июнь','уфимец','стерлитамак','уфимский','житель','женщина','мужчина','геленджик','город','оренбург','томич','область','томский','томск','ставрополье', 'пятигорск', 'ставрополь', 'кисловодск', 'ставрополец', 'невинномысск', 'ессентуки',  'ставропольский', 'край', 'краснодар', 'сочи', 'кубань', 'краснодарский', 'новороссийск', 'башкортостан', 'башкирия', 'уфа', 'пашковский','vip','экипажпассажир','сегодня','другой','многий','вид','день','местный','ваш','качественно','качественный','профессиональный','каневский','станица','район','услуга','другой','ответственный','подъесть','район','гкраснодар','люба','частник','армавир','юмр','анапа','брюховецкий','тимашевск','предлагаться','предлагать','работа','работать','павловский','кущевский','устьлабинск','ленинградский','староминский','гарантия','любой','тип','адлер','год','аванс','частично','пашковка','чмр','сочиадлер','ккб','быстро','недорого','мастер','после','ёмкость','юфо','скорый','дорого','качество','опыт','опытный','круглосуточно','выходной','срочный','срочно','александр','цена','предлагать','принимать','делать','бесплатно','бесплатный','выполнить','выполнять','любой','сделать','предоплата','средний','белореченск','сдавать','сдать','кропоткин','крым','россия','выезд','лазаревский','опыт','минь','кмр','ейск','юмр']

def avitoMagic(text):
	vectorizer = CountVectorizer(charset='koi8r', stop_words=stop_words)

	vectorized_text = vectorizer.fit_transform(text).toarray()
	rf = RandomForestClassifier(n_jobs=-1)
	rf.fit(vectorized_text)

analyzer = pymorphy2.MorphAnalyzer()

def normalizeText(text, filter_regions=False, nouns_adj_only=False, stop_avito=False):
	splittedLines = text.splitlines()
	result = ['' for i in range(len(splittedLines))]
	counter = 0
	for text in splittedLines:
		splittedText = ''.join(ch for ch in text if ch not in set(u'!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~—«0123456789»–')).lower().split()
		splittedText = [w.lower() for w in splittedText if len(w) > 2]
		parsedText = [analyzer.parse(token)[0] for token in splittedText]
		lemmas = []
		for x in parsedText:
			if x.normal_form in stop_words:
				continue
			if filter_regions and x.normal_form in stop_words_regions:
				continue
			if nouns_adj_only and not (('NOUN' in x.tag) or ('ADJF' in x.tag)):
				continue
			if stop_avito and x.normal_form in stop_words_avito:
				continue
			lemmas.append(x.normal_form)
		rr = ""
		for x in lemmas:
			rr+= x + ' '
		result[counter]=rr
		counter+=1
	return '\n'.join(result)


def graphBuilder(all_source_titles_normalized):
	words = all_source_titles_normalized.split()	
	counts = Counter(words)
	bigrams = collections.defaultdict(dict)
	for i in range(0, len(words) - 1):
		if words[i + 1] not in bigrams[words[i]]:
			bigrams[words[i]][words[i+1]] = 1
		else:
			bigrams[words[i]][words[i + 1]] += 1
	relations = collections.defaultdict(dict)
	vertices = {}
	g = Graph(directed=True)
	counter = 0
	print("PERCENTILE IS: ", np.percentile(list(counts.values()), 9))
	for i in range(0, len(words) - 1):
		relations[words[i]][words[i + 1]] = bigrams[words[i]][words[i + 1]] / counts[words[i]]
		if relations[words[i]][words[i+1]] > 0.63 and counts[words[i]] > np.percentile(list(counts.values()), 93) and counts[words[i + 1]] > np.percentile(list(counts.values()), 93):
			if words[i] not in vertices:
				g.add_vertices(1)
				g.vs[counter]["name"] = words[i]
				vertices[words[i]] = counter
				counter += 1
			if words[i + 1] not in vertices:
				g.add_vertices(1)
				g.vs[counter]["name"] = words[i + 1]
				vertices[words[i + 1]] = counter
				counter += 1
			if g.get_eid(vertices[words[i]], vertices[words[i+1]], directed=False, error=False) == -1:
				g.add_edges([(vertices[words[i]], vertices[words[i+1]])])
	g.vs["label"] = g.vs["name"]
	g.vs["color"] = "rgb(224,224,224)"
	g.es["arrow_size"] = 1
	layout = g.layout_kamada_kawai()
	visual_style = {}
	visual_style["vertex_size"] = 1
	visual_style["vertex_label"] = g.vs["name"]
	visual_style["layout"] = layout
	visual_style["bbox"] = (800,800)
	visual_style["margin"] = 50
	visual_style["edge_curved"] = False
	plot(g, "/Users/ilonapapava/Programming/papava.me/internet_portrait/static/pics/graph1.png", **visual_style)
	#graph_draw(g, vertex_text=g.vertex_index, vertex_font_size=18,
	#			output_size=(200, 200), output="two-nodes.png")

import random
# tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),    
#              (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    
#              (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
#              (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
#              (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

# def grey_color_func(word, font_size, position, orientation, random_state=None,
#                     **kwargs):
# 	index = random.randint(0,19)
# 	return "rgb({0},{1},{2})".format(tableau20[index][0], tableau20[index][1], tableau20[index][2])



def wordCloudBuilder(text, path_to_save):
	from wordcloud import WordCloud
	import string
	from functools import partial
	wordcloud = WordCloud(max_font_size=100, background_color='white', prefer_horizontal=1).generate(text)
	plt.figure()
	#plt.imshow(wordcloud.recolor(color_func=grey_color_func, random_state=3),
     #      					interpolation="bilinear")
	plt.imshow(wordcloud, interpolation="bilinear")
	plt.axis("off")
	plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
	plt.savefig(path_to_save)
	plt.close()
	return wordcloud.words_

def movingAverage(all_sources, all_source_titles_normalized):
	left = 0
	right = 0
	counter = 0
	mymap = {}
	listOfWords = set(all_source_titles_normalized.split())
	reversedmap = ['' for i in range(len(listOfWords))]
	for key in listOfWords:
		mymap[key] = counter
		reversedmap[counter] = key
		counter += 1

	titlesPerDay = collections.defaultdict(dict)
	myset=set()
	for i in range(len(all_sources)):
		cur_date = all_sources[i].source_time.date()
		myset.add(cur_date)
		if cur_date not in titlesPerDay:
			titlesPerDay[cur_date] = all_sources[i].source_title
		else:
			titlesPerDay[cur_date] += all_sources[i].source_title

	matrix = np.zeros((len(listOfWords), len(titlesPerDay))).astype(int)
	date_index = 0
	for dates in titlesPerDay:
		currentText = normalizeText(titlesPerDay[dates])
		currentText = currentText.split()
		for word in currentText:
			matrix[mymap[word]][date_index] += 1
		date_index += 1
	res_counter = 0
	maxi = np.amax(matrix, axis=1)
	mini = np.percentile(matrix, 30, axis=1)
	difference = maxi - mini
	indexes = np.argwhere(difference > 30)
	for dates in myset:
		print(dates)
	for value in indexes:
		print(reversedmap[value[0]])
		print(difference[value[0]], maxi[value[0]], mini[value[0]])

if __name__ == '__main__':
	db_file = "/Users/ilonapapava/Programming/papava.me/python/scripts/database.db"
	while True:
		print("If anything running?..")
		conn = sqlite3.connect(db_file)
		c = conn.cursor()
		c.execute("SELECT id, region_name FROM ready_reports WHERE status='PENDING'")
		val = c.fetchone()
		conn.close()
		if bool(val):
			print(val)
			print("Generating report...")
			full_path = generate_report(val[1])
			selected_id = val[0]
			conn = sqlite3.connect(db_file)
			c = conn.cursor()
			c.execute("UPDATE ready_reports SET status='READY', full_path=(?) WHERE id=(?)", (full_path, selected_id))
			conn.commit()
			conn.close()
			continue
		print("Ok, done.. Sleeping for now...")
		time.sleep(120)
	

