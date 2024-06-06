"""WheatRecogBE URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static

from app.views import field, other

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', field.index),
    path('field/create_field', field.create_field),
    path('field/delete_field', field.delete_field),
    path('field/get_all_field', field.get_all_field),
    path('field/get_field/<int:field_id>/', field.get_field),
    path('api/upload_picture', other.upload_picture),
    path('api/upload_video', other.upload_video),
    path('api/upload_file', other.upload_file),
    path('api/upload_file_video', other.upload_file_video),
    path('api/history_query', other.history_query),
    path('api/get_history/<int:history_id>/', other.get_history),
    path('api/delete_history', other.delete_history),
    path('api/test', other.test),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.MEDIA_URL_RAW, document_root=settings.MEDIA_ROOT_RAW)
    urlpatterns += static(settings.MEDIA_URL_OUTCOME, document_root=settings.MEDIA_ROOT_OUTCOME)