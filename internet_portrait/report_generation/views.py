from django.shortcuts import render, redirect

from report_generation.models import Regions, YandexNews, Sources, AvitoAds, ReadyReports
from django.db.models import Count

regionslist = {'22': 'Алтайский край', '28': 'Амурская область', '29': 'Архангельская область', '30': 'Астраханская область', '31': 'Белгородская область', '32': 'Брянская область', '33': 'Владимирская область', '34': 'Волгоградская область', '35': 'Вологодская область', '36': 'Воронежская область', '77': 'г. Москва', '79': 'Еврейская автономная область', '75': 'Забайкальский край', '37': 'Ивановская область', '38': 'Иркутская область', '07': 'Кабардино-Балкарская Республика', '39': 'Калининградская область', '40': 'Калужская область', '41': 'Камчатский край', '09': 'Карачаево-Черкесская Республика', '42': 'Кемеровская область', '43': 'Кировская область', '44': 'Костромская область', '23': 'Краснодарский край', '24': 'Красноярский край', '45': 'Курганская область', '46': 'Курская область', '47': 'Ленинградская область', '48': 'Липецкая область', '49': 'Магаданская область', '50': 'Московская область', '51': 'Мурманская область', '83': 'Ненецкий автономный округ', '52': 'Нижегородская область', '53': 'Новгородская область', '54': 'Новосибирская область', '55': 'Омская область', '56': 'Оренбургская область', '57': 'Орловская область', '58': 'Пензенская область', '59': 'Пермский край', '25': 'Приморский край', '60': 'Псковская область', '01': 'Республика Адыгея', '04': 'Республика Алтай', '02': 'Республика Башкортостан', '03': 'Республика Бурятия', '05': 'Республика Дагестан', '06': 'Республика Ингушетия', '08': 'Республика Калмыкия', '10': 'Республика Карелия', '11': 'Республика Коми', '91': 'Республика Крым', '12': 'Республика Марий Эл', '13': 'Республика Мордовия', '14': 'Республика Саха (Якутия)', '15': 'Республика Северная Осетия - Алания', '16': 'Республика Татарстан', '17': 'Республика Тыва', '19': 'Республика Хакасия', '61': 'Ростовская область', '62': 'Рязанская область', '63': 'Самарская область', '78': 'Санкт-Петербург', '64': 'Саратовская область', '65': 'Сахалинская область', '66': 'Свердловская область', '92': 'Севастополь', '67': 'Смоленская область', '26': 'Ставропольский край', '68': 'Тамбовская область', '69': 'Тверская область', '70': 'Томская область', '71': 'Тульская область', '72': 'Тюменская область', '18': 'Удмуртская Республика', '73': 'Ульяновская область', '27': 'Хабаровский край', '86': 'Ханты-Мансийский автономный округ', '74': 'Челябинская область', '20': 'Чеченская Республика', '21': 'Чувашская Республика', '87': 'Чукотский автономный округ', '89': 'Ямало-Ненецкий автономный округ', '76': 'Ярославская область'}

def index(request):
    return render(request, 'index.html', {'regionslist': regionslist})

def reports_ready(request):
	query_results = ReadyReports.objects.all()
	return render(request, 'reports_ready.html', {'query_results': query_results})

from django.http import HttpResponse
from django.template import Context
from django.template.loader import get_template
from subprocess import Popen, PIPE, CalledProcessError
import tempfile
import os
import time

import sqlite3
import pymorphy2

from igraph import *
import collections
from collections import Counter
import numpy as np

import time
import datetime
from django.http import HttpResponseRedirect

def generate_report(request):
	context = {'content': request.POST.get('region', 'WRONG REGION')}
	from babel.dates import format_datetime
	new_report_to_generate = ReadyReports(region_name=context['content'], date_created=str(format_datetime(datetime.datetime.now(), locale='ru_RU.UTF-8')), status='PENDING')
	new_report_to_generate.save()
	return HttpResponseRedirect('reports_ready.html')
	'''
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
	context['top_news'] = zip([el.title for el in top_news[:5]], [el.num_sources for el in top_news[:5]])
	all_source_titles = ""
	all_sources = all_sources.order_by('source_time')
	for source in all_sources:
	 	all_source_titles += source.source_title
	all_source_titles_normalized = normalizeText(all_source_titles, True, True)
	wordCloudBuilder(all_source_titles_normalized)
	graphBuilder(all_source_titles_normalized)
	#movingAverage(all_sources, all_source_titles_normalized)

    #for tpn in top_news:
    #   	print(tpn.title)
	#all_sources = Sources.objects.all()
	#counter = 0
	#for en in all_entries:
		#if en.news.yandex.name == context['content']:
		#	counter+=1
	#print(counter)
	#context['newsnumber'] = str(counter)
	#c.execute('select count(*) from regions where name=(?)', (context['content'], ))
	#c.execute('select count(*) from regions')
	#context['news_number'] = c.fetchone()[0]
	#conn.close()

	# avito_objects = AvitoAds.objects.filter(avito__name=context['content'])
	# print(len(avito_objects))
	# all_avito_titles=''
	# for i in avito_objects:
	# 	all_avito_titles += i.title
	# print(len(all_avito_titles))
	# all_avito_titles = normalizeText(all_avito_titles)

	# f = open('workfile', 'w')
	# for title in all_avito_titles:
	# 	f.write(title)

	template = get_template('my_latex_template.tex')
	rendered_tpl = template.render(context).encode('utf-8')
	BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
	cur_time = str(time.time())
	with tempfile.TemporaryDirectory() as tempdir:  
		for ppp in range(3):
			try:
				process = Popen(
					['pdflatex', '-output-directory', tempdir],
					stdin=PIPE,
					stdout=PIPE,
					stderr=PIPE,
					cwd=r'/Users/ilonapapava/Programming/papava.me/internet_portrait/static/pics'
				)
				process.communicate(rendered_tpl)
			except e as Error:
				print("FUCK THAT SHIT")
		os.rename(os.path.join(tempdir, 'texput.pdf'), os.path.join(BASE_DIR, 'report_generation/ready_reports/textput' + cur_time + '.pdf'))
	with open(os.path.join(BASE_DIR, 'report_generation/ready_reports/textput' + cur_time + '.pdf'), 'rb') as f:
		pdf = f.read()
	r = HttpResponse(content_type='application/pdf')  
	#r['Content-Disposition'] = 'attachment; filename=texput.pdf'
	r.write(pdf)
	return r

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

stop_words=['изз','под','без', 'а', 'будем', 'будет', 'будете', 'будешь', 'буду', 'будут', 'будучи', 'будь', 'будьте', 'бы', 'был', 'была', 'были', 'было', 'быть', 'в', 'вам', 'вами', 'вас', 'весь', 'во', 'вот', 'все', 'всё', 'всего', 'всей', 'всем', 'всём', 'всеми', 'всему', 'всех', 'всею', 'всея', 'всю', 'вся', 'вы', 'да', 'для', 'до', 'его', 'едим', 'едят', 'ее', 'её', 'ей', 'ел', 'ела', 'ем', 'ему', 'емъ', 'если', 'ест', 'есть', 'ешь', 'еще', 'ещё', 'ею', 'же', 'за', 'и', 'из', 'или', 'им', 'ими', 'имъ', 'их', 'к', 'как', 'кем', 'ко', 'когда', 'кого', 'ком', 'кому', 'комья', 'которая', 'которого', 'которое', 'которой', 'котором', 'которому', 'которою', 'которую', 'которые', 'который', 'которым', 'которыми', 'которых', 'кто', 'меня', 'мне', 'мной', 'мною', 'мог', 'моги', 'могите', 'могла', 'могли', 'могло', 'могу', 'могут', 'мое', 'моё', 'моего', 'моей', 'моем', 'моём', 'моему', 'моею', 'можем', 'можно', 'может', 'можете', 'можешь', 'мои', 'мой', 'моим', 'моими', 'моих', 'мочь', 'мою', 'моя', 'мы', 'на', 'нам', 'нами', 'нас', 'наса', 'наш', 'наша', 'наше', 'нашего', 'нашей', 'нашем', 'нашему', 'нашею', 'наши', 'нашим', 'нашими', 'наших', 'нашу', 'не', 'него', 'нее', 'неё', 'ней', 'нем', 'нём', 'нему', 'нет', 'нею', 'ним', 'ними', 'них', 'но', 'о', 'об', 'один', 'одна', 'одни', 'одним', 'одними', 'одних', 'одно', 'одного', 'одной', 'одном', 'одному', 'одною', 'одну', 'он', 'она', 'оне', 'они', 'оно', 'от', 'по', 'при', 'с', 'сам', 'сама', 'сами', 'самим', 'самими', 'впрочем', 'самих', 'само', 'самого', 'самом', 'самому', 'саму', 'свое', 'своё', 'своего', 'своей', 'своем', 'своём', 'своему', 'своею', 'свои', 'свой', 'своим', 'своими', 'своих', 'свою', 'своя', 'себе', 'себя', 'собой', 'собою', 'та', 'так', 'такая', 'такие', 'таким', 'такими', 'таких', 'такого', 'такое', 'такой', 'таком', 'такому', 'такою', 'такую', 'те', 'тебе', 'тебя', 'тем', 'теми', 'тех', 'то', 'тобой', 'тобою', 'того', 'той', 'только', 'том', 'томах', 'тому', 'тот', 'тою', 'ту', 'ты', 'у', 'уже', 'чего', 'чем', 'чём', 'чему', 'что', 'чтобы', 'эта', 'эти', 'этим', 'этими', 'этих', 'это', 'этого', 'этой', 'этом', 'этому', 'этот', 'этою', 'эту', 'я', 'один', 'два', 'три', 'четыре', 'пять', 'шесть', 'семь', 'восемь', 'девять', 'десять', 'трое']
stop_words_regions = ['человек','республика','башкирский','апрель','май','июнь','уфимец','стерлитамак','уфимский','житель','женщина','мужчина','геленджик','город','оренбург','томич','область','томский','томск','ставрополье', 'пятигорск', 'ставрополь', 'кисловодск', 'ставрополец', 'невинномысск', 'ессентуки',  'ставропольский', 'край', 'краснодар', 'сочи', 'кубань', 'краснодарский', 'новороссийск', 'башкортостан', 'башкирия', 'уфа']
def avitoMagic(text):
	vectorizer = CountVectorizer(charset='koi8r', stop_words=stop_words)

	vectorized_text = vectorizer.fit_transform(text).toarray()
	rf = RandomForestClassifier(n_jobs=-1)
	rf.fit(vectorized_text)

	

analyzer = pymorphy2.MorphAnalyzer()
def normalizeText(text, filter_regions, nouns_adj_only):
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
	#verteces_prop = g.new_vertex_property("string") 
	for i in range(0, len(words) - 1):
		relations[words[i]][words[i + 1]] = bigrams[words[i]][words[i + 1]] / counts[words[i]]
		if relations[words[i]][words[i+1]] > 0.65 and counts[words[i]] > 15 and counts[words[i + 1]] > 15:
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
	g.es["arrow_size"] = 0.7
	layout = g.layout_kamada_kawai()
	visual_style = {}
	visual_style["vertex_size"] = 10
	visual_style["vertex_label"] = g.vs["name"]
	visual_style["layout"] = layout
	visual_style["bbox"] = (500, 500)
	visual_style["margin"] = 15
	plot(g, "/Users/ilonapapava/Programming/papava.me/internet_portrait/static/pics/graph1.png", **visual_style)
	#graph_draw(g, vertex_text=g.vertex_index, vertex_font_size=18,
	#			output_size=(200, 200), output="two-nodes.png")

def wordCloudBuilder(text):
	from wordcloud import WordCloud
	import string
	import matplotlib.pyplot as plt
	wordcloud = WordCloud(max_font_size=100, background_color='white', prefer_horizontal=1).generate(text)
	plt.figure()
	plt.imshow(wordcloud, interpolation="bilinear")
	plt.axis("off")
	plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
	plt.savefig('/Users/ilonapapava/Programming/papava.me/internet_portrait/static/pics/wordcloud.png')

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
'''

