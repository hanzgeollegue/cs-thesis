from django.contrib import admin
from .models import Resume, JobPosting, RankingSession

@admin.register(Resume)
class ResumeAdmin(admin.ModelAdmin):
    list_display = (
        'candidate_id',
        'filename',
        'original_file_path',
        'uploaded_at',
        'processed_at',
        'processing_status',
    )
    readonly_fields = (
        'candidate_id',
        'filename',
        'original_file_path',
        'uploaded_at',
        'processed_at',
        'processing_status',
        'error_message',
        'parsed_data',
        'processed_data',
        'ranking_data',
    )

@admin.register(JobPosting)
class JobPostingAdmin(admin.ModelAdmin):
    list_display = (
        'job_id',
        'title',
        'company',
        'created_at',
        'status',
    )
    readonly_fields = (
        'job_id',
        'title',
        'company',
        'description',
        'requirements',
        'nice_to_have',
        'created_at',
        'status',
        'job_analysis',
    )

@admin.register(RankingSession)
class RankingSessionAdmin(admin.ModelAdmin):
    list_display = (
        'session_id',
        'job_id',
        'candidates_count',
        'created_at',
        'status',
    )
    readonly_fields = (
        'session_id',
        'job_id',
        'job_description',
        'candidates_count',
        'created_at',
        'status',
        'ranking_results',
        'summary',
    )
