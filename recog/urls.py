from django.urls import path
from . import views
urlpatterns = [
    path('',views.home,name='home'),
    path('index',views.index,name='index'),
    path('video_feed',views.video_feed,name='video_feed')
]