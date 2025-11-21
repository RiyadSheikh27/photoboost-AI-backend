from django.apps import AppConfig


class EnhancerConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'enhancer'
    verbose_name = 'Image Enhancer'

    def ready(self):
        """
        Initialize the app when Django starts
        This is a good place to pre-load models if needed
        """
        pass
