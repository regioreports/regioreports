from django.conf.urls import url

from report_generation import views

urlpatterns = [
    url(r'^$', views.index, name='index'),
    url(r'index.html', views.index, name='index'),
    url(r'reports_ready.html', views.reports_ready, name='reports_ready'),
#    url(r'^lessons/(?P<lesson_id>\d+)/$', views.lesson, name='lesson'),
#    url(r'^problems/(?P<problem_id>\d+)/$', views.problem, name='problem'),
    url(r'^generate_report', views.generate_report, name='generate_report'),
]
