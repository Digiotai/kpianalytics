from django.contrib import admin
from django.urls import path, include, re_path
from .views import *
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', testing, name='testing'),
    path('datapx1/', home_page, name="home_page"),
    path('dataprocess', data_process, name='data_process'),
    path('kpi_process/', get_prompt, name="kpi_process"),
    path('mvt/', mvt, name='mvt'),
    path('generate_code/', kpi_code, name="kpi_code"),
    path('kpi_store', kpi_store, name='kpi_store'),
    # path('models', models, {'rf_result': None}, name='models'),
    path(r'models', models, name='models'),
    path('model_predict', model_predict, name='model_predict')
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.MEDIA_URL_1, document_root=settings.MEDIA_ROOT_1)
