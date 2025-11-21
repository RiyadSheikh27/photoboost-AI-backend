"""
URL Configuration for Enhancer App
"""

from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import ImageEnhancementViewSet

router = DefaultRouter()
router.register(r'enhance', ImageEnhancementViewSet, basename='enhance')

urlpatterns = [
    path('', include(router.urls)),
]