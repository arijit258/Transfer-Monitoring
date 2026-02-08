from django.db import models
from django.utils import timezone

class Transformer(models.Model):
    STATUS_CHOICES = [
        ('Healthy', 'Healthy'),
        ('Warning', 'Warning'),
        ('Critical', 'Critical'),
    ]
    
    transformer_id = models.CharField(max_length=20, unique=True)
    name = models.CharField(max_length=200)
    location = models.CharField(max_length=200)
    rating_mva = models.FloatField()
    voltage = models.CharField(max_length=50)
    age_years = models.IntegerField()
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='Healthy')
    health_score = models.IntegerField(default=100)
    scenario = models.CharField(max_length=200, blank=True)
    description = models.TextField(blank=True)
    last_maintenance = models.DateField(null=True, blank=True)
    next_maintenance = models.DateField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.transformer_id} - {self.name}"

    class Meta:
        ordering = ['transformer_id']


class SensorReading(models.Model):
    transformer = models.ForeignKey(Transformer, on_delete=models.CASCADE, related_name='readings')
    timestamp = models.DateTimeField(default=timezone.now)
    top_oil_temp_c = models.FloatField(null=True, blank=True)
    winding_temp_c = models.FloatField(null=True, blank=True)
    ambient_temp_c = models.FloatField(null=True, blank=True)
    load_percent = models.FloatField(null=True, blank=True)
    moisture_ppm = models.FloatField(null=True, blank=True)
    tan_delta_percent = models.FloatField(null=True, blank=True)
    breakdown_voltage_kv = models.FloatField(null=True, blank=True)
    h2_ppm = models.FloatField(null=True, blank=True)
    ch4_ppm = models.FloatField(null=True, blank=True)
    c2h2_ppm = models.FloatField(null=True, blank=True)
    c2h4_ppm = models.FloatField(null=True, blank=True)
    c2h6_ppm = models.FloatField(null=True, blank=True)
    co_ppm = models.FloatField(null=True, blank=True)
    co2_ppm = models.FloatField(null=True, blank=True)

    def __str__(self):
        return f"{self.transformer.transformer_id} - {self.timestamp}"

    class Meta:
        ordering = ['-timestamp']


class RiskFactor(models.Model):
    transformer = models.ForeignKey(Transformer, on_delete=models.CASCADE, related_name='risk_factors')
    description = models.CharField(max_length=500)
    severity = models.CharField(max_length=20, default='Medium')
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.transformer.transformer_id} - {self.description[:50]}"


class Recommendation(models.Model):
    transformer = models.ForeignKey(Transformer, on_delete=models.CASCADE, related_name='recommendations')
    description = models.TextField()
    priority = models.CharField(max_length=20, default='Normal')
    is_completed = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.transformer.transformer_id} - {self.description[:50]}"


class Report(models.Model):
    transformer = models.ForeignKey(Transformer, on_delete=models.CASCADE, related_name='reports')
    title = models.CharField(max_length=200)
    report_type = models.CharField(max_length=50)
    file_path = models.CharField(max_length=500, blank=True)
    content = models.TextField(blank=True)
    generated_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.transformer.transformer_id} - {self.title}"

    class Meta:
        ordering = ['-generated_at']


class ChatSession(models.Model):
    """Chat session for tracking conversations"""
    session_id = models.CharField(max_length=100, unique=True)
    transformer = models.ForeignKey(Transformer, on_delete=models.CASCADE, null=True, blank=True, related_name='chat_sessions')
    model_name = models.CharField(max_length=50, default='phi3')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)

    def __str__(self):
        transformer_id = self.transformer.transformer_id if self.transformer else 'General'
        return f"Session {self.session_id} - {transformer_id}"

    class Meta:
        ordering = ['-updated_at']


class ChatMessage(models.Model):
    """Individual chat message in a session"""
    ROLE_CHOICES = [
        ('user', 'User'),
        ('assistant', 'AI Assistant'),
        ('system', 'System'),
    ]
    
    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE, related_name='messages')
    role = models.CharField(max_length=20, choices=ROLE_CHOICES)
    content = models.TextField()
    model_used = models.CharField(max_length=50, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.session.session_id} - {self.role}: {self.content[:50]}"

    class Meta:
        ordering = ['created_at']
