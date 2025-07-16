"""
Configuration file for RE-FusionX image classification project
Using BLIP2 models with multi-stage classification prompts only
"""
import os
from pathlib import Path
import torch

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw" / "images"
MODELS_BASE_DIR = DATA_DIR / "models"
CACHE_DIR = PROJECT_ROOT / "cache"

# External data paths
IMAGES_ROOT = Path("D:/Phd/images")

# Input/Output files
INPUT_INDEX_FILE = RAW_DATA_DIR / "property_images_index_clean.csv"
OUTPUT_CLASSIFICATION_FILE = RAW_DATA_DIR / "image_classification.csv"
OUTPUT_COMPREHENSIVE_FILE = RAW_DATA_DIR / "image_analysis_comprehensive.csv"
OUTPUT_BEST_IMAGES_FILE = RAW_DATA_DIR / "best_images_per_property.csv"

# BLIP2 models configuration
AVAILABLE_MODELS = {
    "xl": {
        "name": "Salesforce/blip2-flan-t5-xl",
        "batch_size": 128,
        "max_batch_size": 256,
        "expected_vram": 60,
        "model_dir": "blip2-xl"
    },
    "xxl": {
        "name": "Salesforce/blip2-flan-t5-xxl",
        "batch_size": 12,
        "max_batch_size": 16,
        "expected_vram": 28,
        "model_dir": "blip2-xxl"
    }
}

# Default model
DEFAULT_MODEL = "xl"


# Multi-stage classification prompts for BLIP2
CLASSIFICATION_PROMPTS = [
       "Does this photo depict an interior (inside a room or apartment) or an exterior (outside environment)?  If interior, Classify this photo as exactly one of these options: kitchen, living room, bedroom, bathroom, outdoor. Answer with only one of these words.",
       "Describe what you see in this interior space."
]

# Classification settings
TARGET_ROOMS = {"kitchen", "living_room", "bedroom", "bathroom"}

# Room classification keywords for parsing BLIP2 responses
ROOM_KEYWORDS = {
        'kitchen': ['kitchen', 'stove', 'oven', 'refrigerator', 'fridge', 'microwave', 
                   'dishwasher', 'cooking', 'countertop', 'cabinets', 'pantry','mutfak','yemek hazinesi'],
        'bathroom': ['bathroom', 'toilet', 'shower', 'bathtub', 'bath', 'sink', 
                    'vanity',  'towel', 'lavatory','toilet paper','toilet paper holder','toilet paper dispenser','duş','duş kabini','duş kabini with mirror','duş kabini with vanity','duş kabini with vanity and mirror','duş kabini with vanity and mirror and toilet paper dispenser','duş kabini with vanity and mirror and toilet paper dispenser and toilet paper holder','duş kabini with vanity and mirror and toilet paper dispenser and toilet paper holder and toilet paper'],
        'bedroom': ['bedroom', 'bed', 'mattress', 'pillow', 'closet', 'dresser', 
                   'nightstand', 'wardrobe', 'sleeping','vanity with makeup','vanity with mirror','makeup table','makeup','yatak','yatak odası'],
       'living_room': ['living room', 'living', 'couch', 'sofa', 'television', 'tv','sliding door','marble floor',
                       'coffee table', 'armchair', 'entertainment', 'lounge','dining table','dining room','dining area','empty room','oturma','large window','large window with view']
}

# Exclusion keywords for filtering non-room images
EXCLUSION_KEYWORDS = ['elevator', 'balcony', 'hallway', 'staircase', 'exterior', 'outdoor', 
                  'garage', 'basement', 'attic', 'corridor', 'stairs','outside','garden','doorway','gym','walkway','pond','bricks and concrete','concrete','bricks','street with a tree in the middle','tree'
                  ,'street','city','with buildings','cars','restaurant','parked','sidewalk','doorway','railing']

# Quality thresholds
MIN_CONFIDENCE = 0.3
HIGH_CONFIDENCE_THRESHOLD = 0.7
MIN_BLUR_SCORE = 20
MIN_BRIGHTNESS = 50
MAX_BRIGHTNESS = 200
MIN_RESOLUTION = 640 * 480

# Property feature analysis
ENABLE_COMPREHENSIVE_ANALYSIS = True
ANALYSIS_COMPONENTS = {
    'multi_stage_classification': True,  # Primary method
    'technical_quality': True,
    'property_attributes': True
}

# Selection strategy weights
SELECTION_WEIGHTS = {
    'classification_confidence': 0.40,  # Confidence from multi-stage classification
    'technical_quality': 0.30,         # Image technical quality
    'image_order': 0.20,               # Position in property listing
    'response_quality': 0.10           # Quality of BLIP2 responses
}

# Batch processing settings
MAX_RETRIES = 3
TIMEOUT_SECONDS = 30

# Logging
LOG_LEVEL = "INFO"
LOG_FILE = PROJECT_ROOT / "classification.log"

# Image validation settings
SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
MIN_IMAGE_SIZE = (224, 224)

# BLIP2 processing settings
BLIP2_MAX_LENGTH = 50  # Maximum length for BLIP2 responses
BLIP2_MIN_LENGTH = 1   # Minimum length for BLIP2 responses
BLIP2_NUM_BEAMS = 5    # Number of beams for beam search
BLIP2_TEMPERATURE = 1.0 # Temperature for generation


# GPU settings (for BLIP2)
USE_GPU = True
GPU_MEMORY_FRACTION = 0.9

# Processing settings
IMAGE_LOADER_WORKERS = 32
PREPROCESSING_WORKERS = 24
PREFETCH_FACTOR = 16

# Processing optimizations
ENABLE_BATCH_PREPROCESSING = True  # معالجة مسبقة للدفعات
DYNAMIC_BATCH_SIZE = True  # تعديل حجم الدفعة ديناميكياً

# GPU Optimizations
ENABLE_MIXED_PRECISION = True
ENABLE_TORCH_COMPILE = True
PIN_MEMORY = True
USE_FLASH_ATTENTION = True  # جديد

# Caching
ENABLE_IMAGE_CACHE = True
CACHE_SIZE_GB = 20  # زيادة من 10
ENABLE_RESPONSE_CACHE = True  # جديد - تخزين نتائج BLIP2
CACHE_EXPIRY_DAYS = 7
# إضافة أوزان للكلمات المفتاحية
KEYWORD_WEIGHTS = {
    'strong': 3.0,    # كلمات قوية جداً مثل "bedroom", "kitchen"
    'medium': 1.5,    # كلمات متوسطة مثل "couch", "stove"
    'weak': 0.5       # كلمات ضعيفة مثل "window", "floor"
}

# حد أدنى للثقة عند وجود كلمات مفتاحية قوية
MIN_CONFIDENCE_WITH_STRONG_KEYWORDS = 0.75

