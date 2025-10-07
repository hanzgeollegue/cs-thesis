from django.db import models
from django.utils import timezone
import json

class Resume(models.Model):
    candidate_id = models.CharField(max_length=255, default='legacy')
    filename = models.CharField(max_length=255, default='unknown')
    original_file_path = models.CharField(max_length=500, default='unknown')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    processed_at = models.DateTimeField(null=True, blank=True)
    processing_status = models.CharField(max_length=20, default='pending')
    error_message = models.TextField(blank=True, null=True)
    
    # JSON fields stored as text for SQLite compatibility
    parsed_data_json = models.TextField(default='{}')
    processed_data_json = models.TextField(default='{}')
    ranking_data_json = models.TextField(default='{}')

    class Meta:
        db_table = 'resumes'
    
    @property
    def parsed_data(self):
        """Get parsed_data as Python dict"""
        try:
            return json.loads(str(self.parsed_data_json)) if self.parsed_data_json else {}
        except json.JSONDecodeError:
            return {}
    
    @parsed_data.setter
    def parsed_data(self, value):
        """Set parsed_data from Python dict"""
        self.parsed_data_json = json.dumps(value) if value else '{}'
    
    @property
    def processed_data(self):
        """Get processed_data as Python dict"""
        try:
            return json.loads(str(self.processed_data_json)) if self.processed_data_json else {}
        except json.JSONDecodeError:
            return {}
    
    @processed_data.setter
    def processed_data(self, value):
        """Set processed_data from Python dict"""
        self.processed_data_json = json.dumps(value) if value else '{}'
    
    @property
    def ranking_data(self):
        """Get ranking_data as Python dict"""
        try:
            return json.loads(str(self.ranking_data_json)) if self.ranking_data_json else {}
        except json.JSONDecodeError:
            return {}
    
    @ranking_data.setter
    def ranking_data(self, value):
        """Set ranking_data from Python dict"""
        self.ranking_data_json = json.dumps(value) if value else '{}'

class JobPosting(models.Model):
    job_id = models.CharField(max_length=255, unique=True, default='legacy')
    title = models.CharField(max_length=255, default='unknown')
    company = models.CharField(max_length=255, default='unknown')
    description = models.TextField(default='')
    
    # JSON fields stored as text for SQLite compatibility
    requirements_json = models.TextField(default='[]')
    nice_to_have_json = models.TextField(default='[]')
    job_analysis_json = models.TextField(default='{}')
    
    created_at = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=20, default='active')

    class Meta:
        db_table = 'job_postings'
    
    @property
    def requirements(self):
        """Get requirements as Python list"""
        try:
            return json.loads(self.requirements_json) if self.requirements_json else []
        except json.JSONDecodeError:
            return []
    
    @requirements.setter
    def requirements(self, value):
        """Set requirements from Python list"""
        self.requirements_json = json.dumps(value) if value else '[]'
    
    @property
    def nice_to_have(self):
        """Get nice_to_have as Python list"""
        try:
            return json.loads(self.nice_to_have_json) if self.nice_to_have_json else []
        except json.JSONDecodeError:
            return []
    
    @nice_to_have.setter
    def nice_to_have(self, value):
        """Set nice_to_have from Python list"""
        self.nice_to_have_json = json.dumps(value) if value else '[]'
    
    @property
    def job_analysis(self):
        """Get job_analysis as Python dict"""
        try:
            return json.loads(self.job_analysis_json) if self.job_analysis_json else {}
        except json.JSONDecodeError:
            return {}
    
    @job_analysis.setter
    def job_analysis(self, value):
        """Set job_analysis from Python dict"""
        self.job_analysis_json = json.dumps(value) if value else '{}'

class RankingSession(models.Model):
    session_id = models.CharField(max_length=255, unique=True, default='legacy')
    job_id = models.CharField(max_length=255, default='legacy')
    job_description = models.TextField(default='')
    candidates_count = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=20, default='processing')
    
    # JSON fields stored as text for SQLite compatibility
    ranking_results_json = models.TextField(default='[]')
    summary_json = models.TextField(default='{}')

    class Meta:
        db_table = 'ranking_sessions'
    
    @property
    def ranking_results(self):
        """Get ranking_results as Python list"""
        try:
            return json.loads(self.ranking_results_json) if self.ranking_results_json else []
        except json.JSONDecodeError:
            return []
    
    @ranking_results.setter
    def ranking_results(self, value):
        """Set ranking_results from Python list"""
        self.ranking_results_json = json.dumps(value) if value else '[]'
    
    @property
    def summary(self):
        """Get summary as Python dict"""
        try:
            return json.loads(self.summary_json) if self.summary_json else {}
        except json.JSONDecodeError:
            return {}
    
    @summary.setter
    def summary(self, value):
        """Set summary from Python dict"""
        self.summary_json = json.dumps(value) if value else '{}'
