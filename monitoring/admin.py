from django.contrib import admin
from .models import Transformer, SensorReading, RiskFactor, Recommendation, Report

@admin.register(Transformer)
class TransformerAdmin(admin.ModelAdmin):
    list_display = ['transformer_id', 'name', 'status', 'health_score', 'scenario', 'location']
    list_filter = ['status', 'scenario']
    search_fields = ['transformer_id', 'name', 'location']

@admin.register(SensorReading)
class SensorReadingAdmin(admin.ModelAdmin):
    list_display = ['transformer', 'timestamp', 'top_oil_temp_c', 'load_percent', 'h2_ppm']
    list_filter = ['transformer']

@admin.register(RiskFactor)
class RiskFactorAdmin(admin.ModelAdmin):
    list_display = ['transformer', 'description', 'severity']

@admin.register(Recommendation)
class RecommendationAdmin(admin.ModelAdmin):
    list_display = ['transformer', 'description', 'priority', 'is_completed']

@admin.register(Report)
class ReportAdmin(admin.ModelAdmin):
    list_display = ['transformer', 'title', 'report_type', 'generated_at']
