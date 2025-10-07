"""
URL configuration for resume_reviewer project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
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
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from resume_processor import views

urlpatterns = [
    path('', views.home, name='home'),
    path('jd/', views.jd_new, name='jd_new'),
    path('upload/', views.resume_upload, name='resume_upload'),
    path('debug-upload/', views.debug_upload, name='debug_upload'),
    path('batch-progress/<str:task_id>/', views.batch_progress, name='batch_progress'),
    path('processing/', views.processing_status, name='processing_status'),
    path('results/', views.ranking_list, name='ranking_list'),
    path('results/export.json', views.export_results_json, name='export_results_json'),
    path('results/export.csv', views.export_results_csv, name='export_results_csv'),
    path('candidate/<str:candidate_id>/', views.candidate_detail, name='candidate_detail'),
    path('compare/', views.candidate_compare, name='candidate_compare'),
    path('admin/', admin.site.urls),
    path('resume/', include('resume_processor.urls')),
]

# Serve media files during development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
