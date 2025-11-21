"""
DRF Serializers for Image Enhancement
"""

from rest_framework import serializers
from .models import ImageEnhancement


class ImageEnhancementSerializer(serializers.ModelSerializer):
    """Serializer for image enhancement with all parameters"""

    class Meta:
        model = ImageEnhancement
        fields = [
            'id',
            'original_image',
            'enhanced_image',
            'status',
            'error_message',
            'prompt',
            'negative_prompt',
            'seed',
            'upscale_factor',
            'controlnet_scale',
            'controlnet_decay',
            'condition_scale',
            'tile_width',
            'tile_height',
            'denoise_strength',
            'num_inference_steps',
            'solver',
            'created_at',
            'updated_at',
            'processing_time',
        ]
        read_only_fields = ['id', 'enhanced_image', 'status', 'error_message',
                           'created_at', 'updated_at', 'processing_time']


class ImageEnhancementCreateSerializer(serializers.ModelSerializer):
    """Simplified serializer for creating enhancement requests"""

    class Meta:
        model = ImageEnhancement
        fields = [
            'original_image',
            'prompt',
            'negative_prompt',
            'seed',
            'upscale_factor',
            'controlnet_scale',
            'controlnet_decay',
            'condition_scale',
            'tile_width',
            'tile_height',
            'denoise_strength',
            'num_inference_steps',
            'solver',
        ]


class ImageEnhancementResponseSerializer(serializers.ModelSerializer):
    """Serializer for response - only essential fields"""

    class Meta:
        model = ImageEnhancement
        fields = [
            'id',
            'original_image',
            'enhanced_image',
            'status',
            'error_message',
            'created_at',
            'processing_time',
        ]
        read_only_fields = fields