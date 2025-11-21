"""
Django Admin Configuration for Image Enhancement
"""

from django.contrib import admin
from django.utils.html import format_html
from .models import ImageEnhancement


@admin.register(ImageEnhancement)
class ImageEnhancementAdmin(admin.ModelAdmin):
    list_display = [
        'id',
        'status',
        'image_preview',
        'upscale_factor',
        'processing_time',
        'created_at',
    ]
    list_filter = ['status', 'created_at', 'solver']
    search_fields = ['id', 'prompt']
    readonly_fields = [
        'id',
        'status',
        'enhanced_image',
        'error_message',
        'processing_time',
        'created_at',
        'updated_at',
        'image_preview',
        'enhanced_preview',
    ]

    fieldsets = (
        ('Status', {
            'fields': ('id', 'status', 'error_message', 'processing_time')
        }),
        ('Images', {
            'fields': ('original_image', 'image_preview', 'enhanced_image', 'enhanced_preview')
        }),
        ('Enhancement Parameters', {
            'fields': (
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
            ),
            'classes': ('collapse',)
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )

    def image_preview(self, obj):
        if obj.original_image:
            return format_html(
                '<img src="{}" style="max-width: 200px; max-height: 200px;" />',
                obj.original_image.url
            )
        return "No image"
    image_preview.short_description = "Original Preview"

    def enhanced_preview(self, obj):
        if obj.enhanced_image:
            return format_html(
                '<img src="{}" style="max-width: 200px; max-height: 200px;" />',
                obj.enhanced_image.url
            )
        return "Not enhanced yet"
    enhanced_preview.short_description = "Enhanced Preview"