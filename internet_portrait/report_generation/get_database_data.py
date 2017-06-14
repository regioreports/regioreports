# -*- coding: utf-8 -*-
import os, django,sys
sys.path.append('/Users/ilonapapava/Programming/papava.me/internet_portrait')
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "internet_portrait.settings")
django.setup()
from report_generation.models import Regions, YandexNews, Sources, AvitoAds

from async_task import normalizeText

avito_objects = AvitoAds.objects.filter(avito__name='Томская область')
all_avito_titles=''
for i in avito_objects:
	all_avito_titles += i.title
all_avito_titles = normalizeText(all_avito_titles)
#print(all_avito_titles)
f = open('workfile', 'w')
for title in all_avito_titles:
	f.write(title)