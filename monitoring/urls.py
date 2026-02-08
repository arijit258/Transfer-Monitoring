from django.urls import path
from . import views

app_name = 'monitoring'

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('transformers/', views.transformer_list, name='transformer_list'),
    path('transformer/<str:transformer_id>/', views.transformer_detail, name='transformer_detail'),
    path('chat/', views.chat_view, name='chat'),
    path('chat/<str:transformer_id>/', views.transformer_chat, name='transformer_chat'),
    
    # API endpoints
    path('api/transformers/', views.api_transformers, name='api_transformers'),
    path('api/transformers/<str:transformer_id>/readings/', views.api_transformer_readings, name='api_readings'),
    path('api/stats/', views.api_dashboard_stats, name='api_stats'),
    path('api/upload/', views.upload_data, name='upload_data'),
    path('api/chat/', views.api_chat, name='api_chat'),
    path('api/chat/history/clear/', views.api_clear_chat, name='api_clear_chat'),
    path('api/sample-questions/', views.api_sample_questions, name='api_sample_questions'),
    path('api/models/', views.api_models, name='api_models'),
    path('api/chart-data/<str:transformer_id>/', views.api_chart_data, name='api_chart_data'),
]
