import json
import os
from django.conf import settings
from django.core.management.base import BaseCommand
from monitoring.models import Transformer, SensorReading, RiskFactor, Recommendation, Report

class Command(BaseCommand):
    help = 'Load transformer scenarios from JSON file'

    def handle(self, *args, **options):
        json_path = os.path.join(settings.BASE_DIR, 'data_source', 'scenarios', 'transformers_scenarios.json')
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        for t_data in data['transformers']:
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
            
            action = 'Created' if created else 'Updated'
            self.stdout.write(f'{action}: {transformer.transformer_id} - {transformer.name}')
        
        self.stdout.write(self.style.SUCCESS(f'Successfully loaded {len(data["transformers"])} transformers'))
