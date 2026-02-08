"""
Management command to load transformer data from JSON scenario file
"""
import json
from pathlib import Path
from django.core.management.base import BaseCommand
from monitoring.models import Transformer, SensorReading, RiskFactor, Recommendation


class Command(BaseCommand):
    help = 'Load transformer data from scenarios JSON file'
    
    def handle(self, *args, **options):
        base_dir = Path(__file__).resolve().parent.parent.parent.parent
        scenarios_file = base_dir / 'data_source' / 'scenarios' / 'transformers_scenarios.json'
        
        if not scenarios_file.exists():
            self.stderr.write(self.style.ERROR(f'Scenarios file not found: {scenarios_file}'))
            return
        
        with open(scenarios_file, 'r') as f:
            data = json.load(f)
        
        transformers_data = data.get('transformers', [])
        
        for t_data in transformers_data:
            self.stdout.write(f"Processing {t_data.get('id')}...")
            
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
            
            # Add sensor readings
            readings = t_data.get('latest_readings', {})
            if readings:
                SensorReading.objects.create(transformer=transformer, **readings)
            
            # Add risk factors
            for risk in t_data.get('risk_factors', []):
                RiskFactor.objects.get_or_create(
                    transformer=transformer, 
                    description=risk
                )
            
            # Add recommendations
            for rec in t_data.get('recommendations', []):
                Recommendation.objects.get_or_create(
                    transformer=transformer, 
                    description=rec
                )
            
            action = 'Created' if created else 'Updated'
            self.stdout.write(self.style.SUCCESS(f'{action}: {transformer.transformer_id}'))
        
        self.stdout.write(self.style.SUCCESS(f'\nLoaded {len(transformers_data)} transformers successfully!'))
