from django.urls import path
from . import views


urlpatterns = [
    path('hello/', views.say_hello),
    path('upload_image/', views.upload_image)
]