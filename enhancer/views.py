"""
DRF Views for Image Enhancement API
"""

import time
import torch
from io import BytesIO
from PIL import Image
from django.core.files.base import ContentFile
from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from refiners.foundationals.latent_diffusion import solvers

from .models import ImageEnhancement
from .serializers import (
    ImageEnhancementSerializer,
    ImageEnhancementCreateSerializer,
    ImageEnhancementResponseSerializer
)
from .ml_models.model_loader import model_loader


class ImageEnhancementViewSet(viewsets.ModelViewSet):
    """
    ViewSet for Image Enhancement

    Endpoints:
    - POST /api/enhance/ - Upload and enhance an image
    - GET /api/enhance/{id}/ - Get enhancement status and result
    - GET /api/enhance/ - List all enhancements
    """

    queryset = ImageEnhancement.objects.all()
    serializer_class = ImageEnhancementSerializer
    parser_classes = (MultiPartParser, FormParser)

    def get_serializer_class(self):
        if self.action == 'create':
            return ImageEnhancementCreateSerializer
        elif self.action in ['retrieve', 'list']:
            return ImageEnhancementResponseSerializer
        return ImageEnhancementSerializer

    def create(self, request, *args, **kwargs):
        """
        Upload and enhance an image

        Request:
        - original_image: Image file (required)
        - prompt: Text prompt (optional, default: "masterpiece, best quality, highres")
        - negative_prompt: Negative prompt (optional)
        - seed: Random seed (optional, default: 42)
        - upscale_factor: Upscale factor (optional, default: 2.0)
        - And other enhancement parameters...

        Response:
        - id: Enhancement ID
        - status: Processing status
        - original_image: URL to original image
        - enhanced_image: URL to enhanced image (once completed)
        """
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        # Create the enhancement record
        enhancement = serializer.save()

        try:
            # Process the image
            self._process_enhancement(enhancement)

            # Return the result
            response_serializer = ImageEnhancementResponseSerializer(enhancement)
            return Response(
                response_serializer.data,
                status=status.HTTP_201_CREATED
            )

        except Exception as e:
            # Update status to failed
            enhancement.status = 'failed'
            enhancement.error_message = str(e)
            enhancement.save()

            return Response(
                {
                    'error': 'Image enhancement failed',
                    'detail': str(e),
                    'id': str(enhancement.id)
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    def _process_enhancement(self, enhancement: ImageEnhancement):
        """Process the image enhancement"""
        start_time = time.time()

        # Update status to processing
        enhancement.status = 'processing'
        enhancement.save()

        # Load the enhancer model
        enhancer = model_loader.get_enhancer()
        device = enhancer.device

        # Load and prepare the image
        input_image = Image.open(enhancement.original_image.path)

        # Resize if needed to avoid VRAM issues
        side_size = min(input_image.size)
        if side_size > 768:
            scale = 768 / side_size
            new_size = (int(input_image.width * scale), int(input_image.height * scale))
            resized_image = input_image.resize(new_size, resample=Image.Resampling.LANCZOS)
        else:
            resized_image = input_image

        # Get solver type
        solver_type = getattr(solvers, enhancement.solver)

        # Create generator
        generator = torch.Generator(device=device)
        generator.manual_seed(enhancement.seed)

        # Enhance the image
        enhanced_image = enhancer.upscale(
            image=resized_image,
            prompt=enhancement.prompt,
            negative_prompt=enhancement.negative_prompt,
            upscale_factor=enhancement.upscale_factor,
            controlnet_scale=enhancement.controlnet_scale,
            controlnet_scale_decay=enhancement.controlnet_decay,
            condition_scale=enhancement.condition_scale,
            tile_size=(enhancement.tile_height, enhancement.tile_width),
            denoise_strength=enhancement.denoise_strength,
            num_inference_steps=enhancement.num_inference_steps,
            loras_scale={"more_details": 0.5, "sdxl_render": 1.0},
            solver_type=solver_type,
            generator=generator,
        )

        # Save the enhanced image
        output_buffer = BytesIO()
        enhanced_image.save(output_buffer, format='PNG')
        output_buffer.seek(0)

        filename = f"enhanced_{enhancement.id}.png"
        enhancement.enhanced_image.save(
            filename,
            ContentFile(output_buffer.read()),
            save=False
        )

        # Update status
        enhancement.status = 'completed'
        enhancement.processing_time = time.time() - start_time
        enhancement.save()

    @action(detail=True, methods=['get'])
    def download(self, request, pk=None):
        """
        Download the enhanced image

        GET /api/enhance/{id}/download/
        """
        enhancement = self.get_object()

        if enhancement.status != 'completed':
            return Response(
                {'error': 'Enhancement not completed yet'},
                status=status.HTTP_400_BAD_REQUEST
            )

        if not enhancement.enhanced_image:
            return Response(
                {'error': 'Enhanced image not available'},
                status=status.HTTP_404_NOT_FOUND
            )

        # Return the image URL
        return Response({
            'enhanced_image_url': request.build_absolute_uri(enhancement.enhanced_image.url)
        })

    @action(detail=False, methods=['post'])
    def enhance_sync(self, request):
        """
        Synchronous enhancement - returns enhanced image immediately
        This is a simple endpoint that processes the image and returns it directly

        POST /api/enhance/enhance_sync/
        """
        serializer = ImageEnhancementCreateSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        enhancement = serializer.save()

        try:
            self._process_enhancement(enhancement)
            response_serializer = ImageEnhancementResponseSerializer(enhancement)
            return Response(response_serializer.data)

        except Exception as e:
            enhancement.status = 'failed'
            enhancement.error_message = str(e)
            enhancement.save()

            return Response(
                {
                    'error': 'Image enhancement failed',
                    'detail': str(e)
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )