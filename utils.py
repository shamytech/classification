"""
Utility functions for the image classification project utils.py
Enhanced with comprehensive analysis utilities and multi-stage classification support
"""
import os
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from PIL import Image
import cv2
from sklearn.metrics.pairwise import cosine_similarity
import logging
import hashlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


def ensure_directories():
    """Create necessary directories if they don't exist"""
    from config import MODELS_BASE_DIR, RAW_DATA_DIR, CACHE_DIR
    
    for directory in [MODELS_BASE_DIR, RAW_DATA_DIR, CACHE_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def load_property_index(file_path: Path) -> pd.DataFrame:
    """Load property images index from CSV"""
    return pd.read_csv(file_path, dtype={'PropertyID': str})


def load_existing_analysis(file_path: Path) -> Dict[str, List[Dict]]:
    """Load existing comprehensive analysis to avoid reprocessing"""
    analysis = {}
    
    if file_path.exists():
        df = pd.read_csv(file_path, dtype={'PropertyID': str})
        for _, row in df.iterrows():
            property_id = str(row['PropertyID'])
            if property_id not in analysis:
                analysis[property_id] = []
            analysis[property_id].append(row.to_dict())
    
    return analysis


def get_last_processed_property(file_path: Path) -> Optional[str]:
    """Get the last processed property ID from analysis file"""
    if not file_path.exists():
        return None
    
    df = pd.read_csv(file_path, dtype={'PropertyID': str})
    if len(df) == 0:
        return None
    
    return str(df.iloc[-1]['PropertyID'])

def extract_room_type_from_description(description: str, prompt_index: int = None) -> Tuple[str, float, str]:
    """Extract room type from description using keyword matching"""
    from config import ROOM_KEYWORDS, EXCLUSION_KEYWORDS
    
    description_lower = description.lower()
    # معالجة خاصة للسؤال الأول (التصنيف المباشر)
    if prompt_index == 0:
        # إذا كانت الإجابة كلمة واحدة من التصنيفات المباشرة
        clean_response = description_lower.strip()
        
        # معالجة "indoor" منفصلة
        if clean_response == 'indoor':
            return 'unknown', 0.0, "Indoor only - need room type"
        
        # معالجة أنواع الغرف المباشرة
        if clean_response in ['bedroom', 'kitchen', 'bathroom']:
            return clean_response, 0.7, "Direct room classification"
        elif clean_response in ['living room', 'living']:
            return 'living_room', 0.7, "Direct room classification"
        elif clean_response == 'outdoor':
            return 'other', 0.95, "Outdoor detected"
       
    # معالجة خاصة للسؤال الثاني (الوصف التفصيلي)    
    elif prompt_index == 1:
        # فحص المؤشرات القوية على أنها ليست غرفة
        strong_exclusions = [
            ('doorway', 'door way', 'entrance door', 'main door', 'front door'),
            ('hallway', 'hall way', 'corridor', 'passage'),
            ('staircase', 'stairs', 'stairway', 'steps'),
            ('elevator', 'lift'),
            ('lobby', 'foyer', 'entrance hall', 'vestibule'),
            ('balcony', 'terrace', 'patio'),
            ('garage', 'parking', 'carport'),
            ('exterior', 'outside', 'outdoor', 'facade')
        ]
        
        for exclusion_group in strong_exclusions:
            if any(exc in description_lower for exc in exclusion_group):
                # تحقق إضافي - هل يحتوي على عناصر غرفة واضحة؟
                room_indicators = [
                    'bed', 'sofa', 'couch', 'table', 'chair', 'desk',
                    'kitchen', 'stove', 'refrigerator', 'sink',
                    'toilet', 'shower', 'bathtub'
                ]
                
                if not any(indicator in description_lower for indicator in room_indicators):
                    matched = next(exc for exc in exclusion_group if exc in description_lower)
                    return 'excluded', 0.95, f"Strong exclusion: {matched}"
    
    # Check for outdoor first
    if any(word in description_lower for word in ['outdoor', 'outside', 'exterior']):
        return 'other', 0.95, "Outdoor scene detected"
    
    # Check general exclusions
    for exclusion in EXCLUSION_KEYWORDS:
        if exclusion in description_lower:
            # السماح بالاستثناءات إذا كانت هناك عناصر غرفة قوية
            strong_room_words = ['bedroom', 'kitchen', 'bathroom', 'living room']
            if not any(room in description_lower for room in strong_room_words):
                return 'excluded', 0.9, f"Excluded: {exclusion}"
    
    # Calculate scores for each room type
    scores = defaultdict(float)
    matched_keywords = defaultdict(list)
    
    # كلمات مفتاحية قوية جداً
    strong_keywords = {
        'kitchen': ['kitchen', 'stove', 'oven', 'refrigerator', 'fridge', 'dishwasher', 'cooking'],
        'bathroom': ['bathroom', 'toilet', 'shower', 'bathtub', 'bath tub', 'sink', 'basin'],
        'bedroom': ['bedroom', 'bed ', ' bed', 'mattress', 'pillow', 'nightstand', 'night stand'],
        'living_room': ['living room', 'livingroom', 'couch', 'sofa', 'television', 'tv', 'coffee table']
    }
    
    # فحص الكلمات القوية
    for room_type, keywords in strong_keywords.items():
        for keyword in keywords:
            if keyword in description_lower:
                scores[room_type] += 3.0
                matched_keywords[room_type].append(f"STRONG:{keyword}")
    
    # فحص الكلمات العادية
    for room_type, keywords in ROOM_KEYWORDS.items():
        for keyword in keywords:
            if keyword in description_lower and keyword not in strong_keywords.get(room_type, []):
                scores[room_type] += 1.0
                matched_keywords[room_type].append(keyword)
    
    # معالجة الغرف الفارغة
    if not scores:
        if 'empty' in description_lower or 'vacant' in description_lower:
            if any(word in description_lower for word in ['large', 'spacious', 'big']):
                return 'living_room', 0.3, "Empty large room"
            elif any(word in description_lower for word in ['small', 'tiny']):
                return 'bedroom', 0.3, "Empty small room"
            else:
                return 'unknown', 0.2, "Empty room - type unclear"
    
    # Select best match
    if scores:
        best_room = max(scores.items(), key=lambda x: x[1])
        room_type = best_room[0]
        score = best_room[1]
        
        # حساب الثقة
        strong_matches = [k for k in matched_keywords[room_type] if k.startswith("STRONG:")]
        
        if len(strong_matches) >= 2:
            confidence = 0.95  # ثقة عالية جداً مع كلمات قوية متعددة
        elif len(strong_matches) == 1:
            confidence = 0.85  # ثقة عالية مع كلمة قوية واحدة
        elif score >= 3:
            confidence = 0.7   # ثقة جيدة مع كلمات متعددة
        else:
            confidence = min(0.3 + (score * 0.2), 0.6)  # ثقة متوسطة
        
        keywords_found = ", ".join(matched_keywords[room_type])
        return room_type, confidence, f"Keywords: {keywords_found}"
    
    return 'unknown', 0.0, "No matching keywords found"

def is_valid_interior_image(image_path: str) -> bool:
    """Check if image is a valid interior photo (محسنة للأداء)"""
    from config import MIN_IMAGE_SIZE, SUPPORTED_FORMATS
    try:
        # Check file extension
        if not any(str(image_path).lower().endswith(ext) for ext in SUPPORTED_FORMATS):
            return False
        # Check if file exists
        if not os.path.exists(image_path):
            return False
        # تحقق سريع من الهيدر فقط
        try:
            with Image.open(image_path) as img:
                img.verify()  # تحقق من الهيدر فقط
        except Exception:
            return False
        # الآن التحميل الكامل والتحقق الفني
        image = cv2.imread(str(image_path))
        if image is None:
            return False
        height, width = image.shape[:2]
        if height < MIN_IMAGE_SIZE[0] or width < MIN_IMAGE_SIZE[1]:
            return False
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_brightness = gray.mean()
        if mean_brightness < 20 or mean_brightness > 235:
            return False
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv, (40, 40, 40), (80, 255, 255))
        green_ratio = np.sum(green_mask) / (height * width * 255)
        if green_ratio > 0.3:
            return False
        blue_mask = cv2.inRange(hsv, (100, 50, 50), (130, 255, 255))
        upper_half = blue_mask[:height//2, :]
        blue_ratio = np.sum(upper_half) / (height//2 * width * 255)
        if blue_ratio > 0.3:
            return False
        return True
    except Exception as e:
        logger.error(f"Error validating image {image_path}: {e}")
        return False


def batch_is_valid_interior_images(image_paths: list, max_workers: int = None) -> list:
    """تحقق متوازي لعدد كبير من الصور - يعيد قائمة من القيم المنطقية بنفس الترتيب"""
    from config import IMAGE_LOADER_WORKERS
    if max_workers is None:
        max_workers = IMAGE_LOADER_WORKERS
    results = [False] * len(image_paths)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {executor.submit(is_valid_interior_image, path): idx for idx, path in enumerate(image_paths)}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception:
                results[idx] = False
    return results


def calculate_consistency_score(image_result: Dict, all_results: List[Dict]) -> float:
    """Calculate how consistent this result is with other images of same property"""
    if len(all_results) <= 1:
        return 1.0
    
    same_type_count = sum(
        1 for r in all_results 
        if r.get('RoomType') == image_result.get('RoomType')
    )
    
    return same_type_count / len(all_results)


def save_comprehensive_analysis(file_path: Path, analysis_data: Dict[str, Any]):
    """Save comprehensive analysis results"""
    file_exists = file_path.exists()
    
    headers = [
        'PropertyID', 'ImageName', 'ImageOrder', 'RoomType', 'Confidence',
        'Resolution', 'FileSize', 'Brightness', 'Contrast', 'BlurScore', 
        'Sharpness', 'Colorfulness', 'OverallQualityScore',
        'ProcessingTime', 'Timestamp', 'ClassificationMethod',
        'ConsistencyScore', 'MultiStageResults', 'BLIP2Response1',
        'BLIP2Response2'
    ]
    
    with open(file_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction='ignore')
        if not file_exists:
            writer.writeheader()
        writer.writerow(analysis_data)


def save_best_images(file_path: Path, property_id: str, best_images: Dict[str, Dict]):
    """Save selected best images for each property"""
    file_exists = file_path.exists()
    
    with open(file_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['PropertyID', 'Kitchen', 'LivingRoom', 'Bedroom', 'Bathroom'])
        
        row = [property_id]
        for room_type in ['kitchen', 'living_room', 'bedroom', 'bathroom']:
            if room_type in best_images:
                row.append(best_images[room_type]['ImageName'])
            else:
                row.append('')
        
        writer.writerow(row)


def analyze_image_quality(image_path: Path) -> Dict[str, float]:
    """Comprehensive technical quality analysis"""
    try:
        # Open with PIL for basic metrics
        pil_image = Image.open(image_path)
        width, height = pil_image.size
        file_size = os.path.getsize(image_path) / 1024  # KB
        
        # Convert to numpy for advanced analysis
        img_array = np.array(pil_image.convert('RGB'))
        
        # Basic metrics
        brightness = np.mean(img_array)
        contrast = np.std(img_array)
        
        # Convert to grayscale for certain metrics
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Blur detection using Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_score = min(laplacian_var / 100, 100)
        
        # Sharpness using gradient magnitude
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        sharpness = np.mean(gradient_magnitude)
        
        # Colorfulness metric
        rg = img_array[:,:,0] - img_array[:,:,1]
        yb = 0.5 * (img_array[:,:,0] + img_array[:,:,1]) - img_array[:,:,2]
        colorfulness = np.sqrt(np.var(rg) + np.var(yb)) + 0.3 * np.sqrt(np.mean(rg**2) + np.mean(yb**2))
        
        # Edge density (indicates detail level)
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.sum(edges > 0) / (width * height)
        
        # Exposure quality
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()
        
        # Check for over/under exposure
        underexposed = np.sum(hist[:50])
        overexposed = np.sum(hist[206:])
        exposure_quality = 1.0 - (underexposed + overexposed)
        
        # Coverage score - how much of the room is visible
        coverage_score = calculate_coverage_score(img_array)
        
        # Clarity score - focus and detail
        clarity_score = calculate_clarity_score(gray)
        
        return {
            'Resolution': f"{width}x{height}",
            'ResolutionScore': min((width * height) / (1920 * 1080), 1.0),
            'FileSize': round(file_size, 2),
            'Brightness': round(brightness, 2),
            'Contrast': round(contrast, 2),
            'BlurScore': round(blur_score, 2),
            'Sharpness': round(sharpness, 2),
            'Colorfulness': round(colorfulness, 2),
            'EdgeDensity': round(edge_density, 4),
            'ExposureQuality': round(exposure_quality, 2),
            'CoverageScore': round(coverage_score, 2),
            'ClarityScore': round(clarity_score, 2)
        }
        
    except Exception as e:
        logger.error(f"Error analyzing image {image_path}: {e}")
        return {
            'Resolution': 'error',
            'ResolutionScore': 0,
            'FileSize': 0,
            'Brightness': 0,
            'Contrast': 0,
            'BlurScore': 0,
            'Sharpness': 0,
            'Colorfulness': 0,
            'EdgeDensity': 0,
            'ExposureQuality': 0,
            'CoverageScore': 0,
            'ClarityScore': 0
        }


def batch_analyze_image_quality(image_paths: list, max_workers: int = None) -> list:
    """تحليل جودة فني متوازي لعدد كبير من الصور - يعيد قائمة من dict بنفس الترتيب"""
    from config import PREPROCESSING_WORKERS
    if max_workers is None:
        max_workers = PREPROCESSING_WORKERS
    results = [None] * len(image_paths)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {executor.submit(analyze_image_quality, path): idx for idx, path in enumerate(image_paths)}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception:
                results[idx] = None
    return results


def calculate_coverage_score(image: np.ndarray) -> float:
    """Calculate how much of the room is visible in the image"""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        edges = cv2.Canny(gray, 50, 150)
        
        # Calculate edge density in different regions
        h, w = edges.shape
        regions = [
            edges[:h//3, :],      # Top
            edges[h//3:2*h//3, :], # Middle
            edges[2*h//3:, :],     # Bottom
            edges[:, :w//3],       # Left
            edges[:, w//3:2*w//3], # Center
            edges[:, 2*w//3:]      # Right
        ]
        
        edge_densities = [np.sum(region) / region.size for region in regions]
        
        # Good coverage means edges in all regions
        coverage = sum(1 for density in edge_densities if density > 0.01) / len(regions)
        
        return coverage
        
    except Exception as e:
        logger.error(f"Error calculating coverage: {e}")
        return 0.5


def calculate_clarity_score(gray_image: np.ndarray) -> float:
    """Calculate image clarity score based on focus and detail"""
    try:
        # Laplacian variance (focus measure)
        laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
        focus_score = np.var(laplacian)
        
        # Gradient magnitude
        grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        gradient_score = np.mean(gradient_magnitude)
        
        # Local contrast
        kernel_size = 15
        local_mean = cv2.blur(gray_image, (kernel_size, kernel_size))
        local_variance = cv2.blur((gray_image - local_mean)**2, (kernel_size, kernel_size))
        contrast_score = np.mean(np.sqrt(local_variance))
        
        # Normalize and combine
        focus_norm = min(focus_score / 1000, 1.0)
        gradient_norm = min(gradient_score / 50, 1.0)
        contrast_norm = min(contrast_score / 50, 1.0)
        
        clarity = (focus_norm + gradient_norm + contrast_norm) / 3
        
        return clarity
        
    except Exception as e:
        logger.error(f"Error calculating clarity: {e}")
        return 0.5


def calculate_overall_quality_score(technical_metrics: Dict, confidence: float, 
                                  response_quality: float = 1.0) -> float:
    """Calculate overall quality score from multiple metrics"""
    weights = {
        'confidence': 0.25,
        'blur': 0.15,
        'exposure': 0.15,
        'sharpness': 0.15,
        'resolution': 0.10,
        'colorfulness': 0.10,
        'coverage': 0.05,
        'response_quality': 0.05
    }
    
    scores = {
        'confidence': confidence,
        'blur': technical_metrics.get('BlurScore', 0) / 100,
        'exposure': technical_metrics.get('ExposureQuality', 0),
        'sharpness': min(technical_metrics.get('Sharpness', 0) / 50, 1.0),
        'resolution': technical_metrics.get('ResolutionScore', 0),
        'colorfulness': min(technical_metrics.get('Colorfulness', 0) / 100, 1.0),
        'coverage': technical_metrics.get('CoverageScore', 0.5),
        'response_quality': response_quality
    }
    
    overall_score = sum(scores[k] * weights[k] for k in weights)
    return round(overall_score * 100, 2)


def select_best_images_per_property(property_images: List[Dict], 
                                   selection_weights: Dict = None) -> Dict[str, Dict]:
    """Select best image for each room type based on comprehensive scores"""
    from config import SELECTION_WEIGHTS
    
    if selection_weights is None:
        selection_weights = SELECTION_WEIGHTS
    
    best_images = {}
    
    # Group by room type
    room_groups = {}
    for img in property_images:
        room_type = img.get('RoomType')
        if room_type and room_type not in ['error', 'unknown', 'excluded', 'outdoor', 'other']:
            if room_type not in room_groups:
                room_groups[room_type] = []
            room_groups[room_type].append(img)
    
    # Calculate consistency scores for all images
    all_images = [img for images in room_groups.values() for img in images]
    for img in all_images:
        img['ConsistencyScore'] = calculate_consistency_score(img, all_images)
    
    # Select best for each room type
    for room_type, images in room_groups.items():
        if images:
            # Calculate composite score for each image
            for img in images:
                composite_score = 0
                
                # Classification confidence from BLIP2
                composite_score += selection_weights.get('classification_confidence', 0.4) * img.get('Confidence', 0)
                
                # Technical quality score
                composite_score += selection_weights.get('technical_quality', 0.3) * img.get('OverallQualityScore', 0) / 100
                
                # Image order (earlier images often better)
                composite_score += selection_weights.get('image_order', 0.2) * (1 / (img.get('ImageOrder', 999) + 1))
                
                # Response quality (based on BLIP2 response completeness)
                response_quality = calculate_response_quality(img)
                composite_score += selection_weights.get('response_quality', 0.1) * response_quality
                
                img['CompositeScore'] = composite_score
            
            # Sort by composite score and select best
            images.sort(key=lambda x: x['CompositeScore'], reverse=True)
            best_images[room_type] = images[0]
    
    return best_images


def calculate_response_quality(image_data: Dict) -> float:
    """Calculate quality score based on BLIP2 responses"""
    quality_score = 0.0
    response_count = 0
    
    # Check each BLIP2 response
    for i in range(1, 3):
        response_key = f'BLIP2Response{i}'
        if response_key in image_data and image_data[response_key]:
            response = str(image_data[response_key])
            if response and response.lower() not in ['none', 'error', 'unknown']:
                response_count += 1
                # Longer, more detailed responses get higher scores
                if len(response) > 50:
                    quality_score += 1.0
                elif len(response) > 20:
                    quality_score += 0.7
                else:
                    quality_score += 0.4
    
    # Normalize by number of responses
    if response_count > 0:
        return quality_score / 2  # Normalize to 0-1 range
    return 0.0


def format_time(seconds: float) -> str:
    """Format seconds into readable time string"""
    return str(timedelta(seconds=int(seconds)))


def calculate_eta(processed: int, total: int, elapsed_time: float) -> str:
    """Calculate estimated time of arrival"""
    if processed == 0:
        return "Calculating..."
    
    rate = processed / elapsed_time
    remaining = total - processed
    eta_seconds = remaining / rate
    
    return format_time(eta_seconds)


# Cache management functions
def get_cache_key(image_path: str, function_name: str) -> str:
    """Generate cache key for an image and function"""
    path_hash = hashlib.md5(str(image_path).encode()).hexdigest()
    return f"{function_name}_{path_hash}"


def load_from_cache(cache_key: str, cache_dir: Path) -> Optional[Dict]:
    """Load result from cache if available and not expired"""
    from config import CACHE_EXPIRY_DAYS
    
    cache_file = cache_dir / f"{cache_key}.json"
    
    if not cache_file.exists():
        return None
    
    try:
        with open(cache_file, 'r') as f:
            cached_data = json.load(f)
        
        # Check expiry
        cached_time = datetime.fromisoformat(cached_data['timestamp'])
        if datetime.now() - cached_time > timedelta(days=CACHE_EXPIRY_DAYS):
            cache_file.unlink()  # Delete expired cache
            return None
        
        return cached_data['data']
        
    except Exception as e:
        logger.error(f"Error loading cache: {e}")
        return None


def save_to_cache(cache_key: str, data: Dict, cache_dir: Path):
    """Save result to cache"""
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / f"{cache_key}.json"
    
    try:
        # Convert numpy types to Python native types
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(convert_numpy_types(item) for item in obj)
            else:
                return obj
        
        # Convert data
        converted_data = convert_numpy_types(data)
        
        cache_data = {
            'timestamp': datetime.now().isoformat(),
            'data': converted_data
        }
        
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f)
            
    except Exception as e:
        logger.error(f"Error saving to cache: {e}")
