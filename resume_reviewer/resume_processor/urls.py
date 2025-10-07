from django.urls import path
from . import views

app_name = 'resume_processor'

urlpatterns = [
    path('upload/', views.upload_resume, name='upload_resume'),
    path('list/', views.resume_list, name='resume_list'),
    path('detail/<int:resume_id>/', views.resume_detail, name='resume_detail'),
    path('structured/<int:resume_id>/', views.resume_structured, name='resume_structured'),
    path('json/<int:resume_id>/', views.resume_json, name='resume_json'),
    path('api/process/<int:resume_id>/', views.process_resume_api, name='process_resume_api'),
    path('delete/<int:resume_id>/', views.delete_resume, name='delete_resume'),
    path('download-json/<int:resume_id>/', views.download_json, name='download_json'),
    path('ranking/', views.ranking_view, name='ranking'),
    path('all-batches/', views.all_batches_view, name='all_batches'),
    path('api/deterministic-ranking/', views.deterministic_ranking_api, name='deterministic_ranking'),
] 