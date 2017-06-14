from final_with_db import parseYandexDate
import datetime

def test_yesterday():
	date = datetime.datetime.now() - datetime.timedelta(1)
	date = date.replace(minute=3, hour=11, second=0, microsecond=0)
	assert str(parseYandexDate('вчера в 11:03')) == str(date)
	date = date.replace(minute=3, hour=8, second=0, microsecond=0)
	assert str(parseYandexDate('вчера в 08:03')) == str(date)


def test_today():
	date = datetime.datetime.now()
	date = date.replace(minute=3, hour=12, second=0, microsecond=0)
	assert str(parseYandexDate('12:03')) == str(date)

def test_months():
	date = datetime.datetime.now()
	date = date.replace(day=11, minute=3, hour=12, second=0, microsecond=0)
	date = date.replace(month=1)
	assert str(parseYandexDate('11 января в 12:03')) == str(date)
	date = date.replace(month=2)
	assert str(parseYandexDate('11 февраля в 12:03')) == str(date)
	date = date.replace(month=3)
	assert str(parseYandexDate('11 марта в 12:03')) == str(date)
	date = date.replace(month=4)
	assert str(parseYandexDate('11 апреля в 12:03')) == str(date)
	date = date.replace(month=5)
	assert str(parseYandexDate('11 мая в 12:03')) == str(date)
	date = date.replace(month=6)
	assert str(parseYandexDate('11 июня в 12:03')) == str(date)
	date = date.replace(month=7)
	assert str(parseYandexDate('11 июля в 12:03')) == str(date)
	date = date.replace(month=8)
	assert str(parseYandexDate('11 августа в 12:03')) == str(date)
	date = date.replace(month=9)
	assert str(parseYandexDate('11 сентября в 12:03')) == str(date)
	date = date.replace(month=10)
	assert str(parseYandexDate('11 октября в 12:03')) == str(date)
	date = date.replace(month=11)
	assert str(parseYandexDate('11 ноября в 12:03')) == str(date)
	date = date.replace(month=12)
	assert str(parseYandexDate('11 декабря в 12:03')) == str(date)

def test_old():
	date = datetime.datetime.now()
	date = date.replace(day=22, month=12, year=2016, minute=40, hour=21, second=0, microsecond=0)
	assert str(parseYandexDate('22.12.16 в 21:40')) == str(date)
	date = date.replace(day=1, month=2, year=2016, minute=33, hour=23, second=0, microsecond=0)
	assert str(parseYandexDate('01.02.16 в 23:33')) == str(date)
	date = date.replace(day=3, month=11, year=2013, minute=0, hour=0, second=0, microsecond=0)
	assert str(parseYandexDate('03.11.13 в 00:00')) == str(date)

test_yesterday()
test_today()
test_months()
test_old()
print("All test passes")