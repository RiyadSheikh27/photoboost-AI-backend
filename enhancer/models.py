"""
Django Models for Image Enhancement
"""

from django.db import models
from django.core.validators import FileExtensionValidator
import uuid


class ImageEnhancement(models.Model):
    """Model to track image enhancement requests"""

    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    original_image = models.ImageField(
        upload_to='originals/%Y/%m/%d/',
        validators=[FileExtensionValidator(allowed_extensions=['jpg', 'jpeg', 'png', 'webp', 'heic'])]
    )
    enhanced_image = models.ImageField(
        upload_to='enhanced/%Y/%m/%d/',
        null=True,
        blank=True
    )
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    error_message = models.TextField(null=True, blank=True)

    # Enhancement parameters
    prompt = models.TextField(default="masterpiece, best quality, highres")
    negative_prompt = models.TextField(default="worst quality, low quality, normal quality")
    seed = models.IntegerField(default=42)
    upscale_factor = models.FloatField(default=2.0)
    controlnet_scale = models.FloatField(default=0.6)
    controlnet_decay = models.FloatField(default=1.0)
    condition_scale = models.IntegerField(default=6)
    tile_width = models.IntegerField(default=112)
    tile_height = models.IntegerField(default=144)
    denoise_strength = models.FloatField(default=0.35)
    num_inference_steps = models.IntegerField(default=18)
    solver = models.CharField(max_length=20, default="DDIM")

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    processing_time = models.FloatField(null=True, blank=True)  # in seconds

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['-created_at']),
            models.Index(fields=['status']),
        ]

    def __str__(self):
        return f"Enhancement {self.id} - {self.status}"