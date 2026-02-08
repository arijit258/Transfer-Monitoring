from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse, FileResponse
from django.views.decorators.csrf import csrf_exempt
from django.db.models import Avg
from .models import Transformer, SensorReading, RiskFactor, Recommendation, Report, ChatSession, ChatMessage
import json
import os
import uuid
from pathlib import Path

# AI imports
try:
    from .ai.rag_engine import get_engine, SAMPLE_QUESTIONS, OllamaClient
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    SAMPLE_QUESTIONS = {}


def dashboard(request):
    transformers = Transformer.objects.all()
    total = transformers.count()
    healthy = transformers.filter(status='Healthy').count()
    warning = transformers.filter(status='Warning').count()
    critical = transformers.filter(status='Critical').count()
    avg_health = transformers.aggregate(avg=Avg('health_score'))['avg'] or 0
    
    context = {
        'transformers': transformers,
        'total_count': total,
        'healthy_count': healthy,
        'warning_count': warning,
        'critical_count': critical,
        'avg_health_score': round(avg_health, 1),
        'ai_available': AI_AVAILABLE,
    }
    return render(request, 'monitoring/dashboard.html', context)


def transformer_detail(request, transformer_id):
    transformer = get_object_or_404(Transformer, transformer_id=transformer_id)
    latest_reading = transformer.readings.first()
    risk_factors = transformer.risk_factors.all()
    recommendations = transformer.recommendations.filter(is_completed=False)
    reports = transformer.reports.all()[:5]
    
    context = {
        'transformer': transformer,
        'latest_reading': latest_reading,
        'risk_factors': risk_factors,
        'recommendations': recommendations,
        'reports': reports,
        'ai_available': AI_AVAILABLE,
    }
    return render(request, 'monitoring/transformer_detail.html', context)


def transformer_list(request):
    status_filter = request.GET.get('status', '')
    transformers = Transformer.objects.all()
    if status_filter:
        transformers = transformers.filter(status=status_filter)
    
    context = {
        'transformers': transformers,
        'status_filter': status_filter,
    }
    return render(request, 'monitoring/transformer_list.html', context)


def chat_view(request):
    """Global chat view"""
    transformers = Transformer.objects.all()
    
    # Get or create a default session for global chat
    session_id = request.session.get('chat_session_id', str(uuid.uuid4()))
    request.session['chat_session_id'] = session_id
    
    session, created = ChatSession.objects.get_or_create(
        session_id=session_id,
        defaults={'model_name': 'phi3'}
    )
    
    # Load recent messages for the session
    messages = list(session.messages.all()[:50])
    
    context = {
        'transformers': transformers,
        'sample_questions': SAMPLE_QUESTIONS,
        'ai_available': AI_AVAILABLE,
        'session_id': session_id,
        'chat_history': messages,
    }
    return render(request, 'monitoring/chat.html', context)


def transformer_chat(request, transformer_id):
    """Transformer-specific chat view"""
    transformer = get_object_or_404(Transformer, transformer_id=transformer_id)
    
    # Get or create a session for this transformer
    session_id = request.session.get(f'chat_session_{transformer_id}', str(uuid.uuid4()))
    request.session[f'chat_session_{transformer_id}'] = session_id
    
    session, created = ChatSession.objects.get_or_create(
        session_id=session_id,
        transformer=transformer,
        defaults={'model_name': 'phi3'}
    )
    
    # Load recent messages for the session
    messages = list(session.messages.all()[:50])
    
    context = {
        'transformer': transformer,
        'sample_questions': SAMPLE_QUESTIONS,
        'ai_available': AI_AVAILABLE,
        'session_id': session_id,
        'chat_history': messages,
    }
    return render(request, 'monitoring/transformer_chat.html', context)


# API Views
def api_transformers(request):
    transformers = Transformer.objects.all().values(
        'transformer_id', 'name', 'location', 'status', 
        'health_score', 'scenario', 'rating_mva', 'voltage'
    )
    return JsonResponse(list(transformers), safe=False)


def api_transformer_readings(request, transformer_id):
    transformer = get_object_or_404(Transformer, transformer_id=transformer_id)
    readings = transformer.readings.all()[:100].values()
    return JsonResponse(list(readings), safe=False)


def api_dashboard_stats(request):
    transformers = Transformer.objects.all()
    stats = {
        'total': transformers.count(),
        'healthy': transformers.filter(status='Healthy').count(),
        'warning': transformers.filter(status='Warning').count(),
        'critical': transformers.filter(status='Critical').count(),
        'avg_health_score': round(transformers.aggregate(avg=Avg('health_score'))['avg'] or 0, 1),
        'transformers': list(transformers.values('transformer_id', 'name', 'status', 'health_score', 'scenario'))
    }
    return JsonResponse(stats)


@csrf_exempt
def upload_data(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            transformers_data = data.get('transformers', [])
            
            for t_data in transformers_data:
                transformer, created = Transformer.objects.update_or_create(
                    transformer_id=t_data['id'],
                    defaults={
                        'name': t_data.get('name', ''),
                        'location': t_data.get('location', ''),
                        'rating_mva': t_data.get('rating_mva', 0),
                        'voltage': t_data.get('voltage', ''),
                        'age_years': t_data.get('age_years', 0),
                        'status': t_data.get('status', 'Healthy'),
                        'health_score': t_data.get('health_score', 100),
                        'scenario': t_data.get('scenario', ''),
                        'description': t_data.get('description', ''),
                    }
                )
                
                readings = t_data.get('latest_readings', {})
                if readings:
                    SensorReading.objects.create(transformer=transformer, **readings)
                
                for risk in t_data.get('risk_factors', []):
                    RiskFactor.objects.get_or_create(transformer=transformer, description=risk)
                
                for rec in t_data.get('recommendations', []):
                    Recommendation.objects.get_or_create(transformer=transformer, description=rec)
            
            return JsonResponse({'status': 'success', 'count': len(transformers_data)})
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=400)
    
    return JsonResponse({'error': 'POST required'}, status=405)


@csrf_exempt
def api_chat(request):
    """API endpoint for chat with AI"""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)
    
    if not AI_AVAILABLE:
        return JsonResponse({
            'error': 'AI module not available. Please install required dependencies.',
            'answer': 'AI features are currently unavailable. Please check server configuration.'
        }, status=503)
    
    try:
        data = json.loads(request.body)
        question = data.get('question', '').strip()
        transformer_id = data.get('transformer_id')
        model = data.get('model', 'phi3')
        session_id = data.get('session_id')
        
        if not question:
            return JsonResponse({'error': 'Question required'}, status=400)
        
        # Get or create session
        session = None
        if session_id:
            try:
                session = ChatSession.objects.get(session_id=session_id)
                session.is_active = True
                session.model_name = model
                session.save()
            except ChatSession.DoesNotExist:
                if transformer_id:
                    transformer = Transformer.objects.get(transformer_id=transformer_id)
                    session = ChatSession.objects.create(
                        session_id=session_id,
                        transformer=transformer,
                        model_name=model
                    )
                else:
                    session = ChatSession.objects.create(
                        session_id=session_id,
                        model_name=model
                    )
        
        # Save user message to database
        if session:
            ChatMessage.objects.create(
                session=session,
                role='user',
                content=question,
                model_used=model
            )
        
        engine = get_engine()
        result = engine.ask(
            question=question,
            transformer_id=transformer_id,
            model=model
        )
        
        # Save AI response to database
        if session:
            ChatMessage.objects.create(
                session=session,
                role='assistant',
                content=result.get('answer', ''),
                model_used=model
            )
        
        return JsonResponse(result)
    
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)
    except Exception as e:
        return JsonResponse({
            'error': str(e),
            'answer': f'Error processing request: {str(e)}'
        }, status=500)


@csrf_exempt
def api_clear_chat(request):
    """Clear chat history"""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)
    
    try:
        data = json.loads(request.body)
        session_id = data.get('session_id')
        
        if session_id:
            # Clear specific session
            ChatMessage.objects.filter(session__session_id=session_id).delete()
            ChatSession.objects.filter(session_id=session_id).delete()
        elif AI_AVAILABLE:
            # Fallback to in-memory clear
            engine = get_engine()
            engine.clear_history()
        
        return JsonResponse({'status': 'success'})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)


def api_sample_questions(request):
    """Get sample questions"""
    return JsonResponse(SAMPLE_QUESTIONS)


def api_models(request):
    """Get available AI models (Ollama and Groq)"""
    if not AI_AVAILABLE:
        return JsonResponse({
            'available': False,
            'ollama': {
                'available': False,
                'models': ['phi3', 'mistral', 'llama3']
            },
            'groq': {
                'available': False,
                'models': ['llama-3.3-70b-versatile', 'llama-3.1-8b-instant', 'mixtral-8x7b-32768', 'gemma-7b-it'],
                'message': 'AI module not available'
            },
            'all_models': {
                'ollama': ['phi3', 'mistral', 'llama3'],
                'groq': ['llama-3.3-70b-versatile', 'llama-3.1-8b-instant', 'mixtral-8x7b-32768', 'gemma-7b-it']
            }
        })
    
    try:
        from .ai.rag_engine import OllamaClient, GroqClient
        
        # Check Ollama
        ollama_client = OllamaClient()
        ollama_available = ollama_client.is_available()
        ollama_models = ollama_client.list_models() if ollama_available else ['phi3', 'mistral', 'llama3']
        
        # Check Groq
        groq_client = GroqClient()
        groq_available = groq_client.is_available()
        groq_models = groq_client.list_models() if groq_available else groq_client.available_models
        
        return JsonResponse({
            'available': ollama_available or groq_available,
            'ollama': {
                'available': ollama_available,
                'models': ollama_models
            },
            'groq': {
                'available': groq_available,
                'models': groq_models,
                'message': 'Groq API configured' if groq_available else 'Groq API key not configured (set GROQ_API_KEY in .env)'
            },
            'all_models': {
                'ollama': ollama_models,
                'groq': groq_models
            }
        })
    except Exception as e:
        return JsonResponse({
            'available': False,
            'ollama': {
                'available': False,
                'models': ['phi3', 'mistral', 'llama3']
            },
            'groq': {
                'available': False,
                'models': ['llama-3.3-70b-versatile', 'llama-3.1-8b-instant', 'mixtral-8x7b-32768', 'gemma-7b-it'],
                'error': str(e)
            },
            'all_models': {
                'ollama': ['phi3', 'mistral', 'llama3'],
                'groq': ['llama-3.3-70b-versatile', 'llama-3.1-8b-instant', 'mixtral-8x7b-32768', 'gemma-7b-it']
            }
        })


def api_chart_data(request, transformer_id):
    """Get chart data for a transformer"""
    import pandas as pd
    
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data_source" / "transformers" / transformer_id.upper()
    
    result = {
        'transformer_id': transformer_id,
        'dga_data': [],
        'oil_data': [],
        'sensor_data': []
    }
    
    # Load DGA history
    dga_file = data_dir / "dga_history.csv"
    if dga_file.exists():
        try:
            df = pd.read_csv(dga_file)
            result['dga_data'] = df.to_dict(orient='records')
        except Exception as e:
            result['dga_error'] = str(e)
    
    # Load oil quality data
    oil_file = data_dir / "oil_quality.csv"
    if oil_file.exists():
        try:
            df = pd.read_csv(oil_file)
            result['oil_data'] = df.to_dict(orient='records')
        except Exception as e:
            result['oil_error'] = str(e)
    
    # Load sensor readings
    sensor_file = data_dir / "sensor_readings.csv"
    if sensor_file.exists():
        try:
            df = pd.read_csv(sensor_file)
            result['sensor_data'] = df.to_dict(orient='records')
        except Exception as e:
            result['sensor_error'] = str(e)
    
    return JsonResponse(result)
