"""
Optimized batch processing with comprehensive analysis batch_processor.py
Using BLIP2 for multi-stage classification and technical analysis
"""
from PIL import Image
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue
import logging
import time
from collections import defaultdict
from transformers import (
    Blip2Processor, 
    Blip2ForConditionalGeneration
)
import torch

from utils import (
    analyze_image_quality,
    calculate_overall_quality_score,
    extract_room_type_from_description,
    is_valid_interior_image,
    calculate_consistency_score,
    get_cache_key,
    load_from_cache,
    save_to_cache,
    calculate_response_quality
)

logger = logging.getLogger(__name__)


class PropertyImageDataset:
    """Custom dataset for efficient batch loading"""
    
    def __init__(self, image_paths: List[Path], transform=None):
        self.image_paths = image_paths
        self.transform = transform
        self.cache = {}
        self.cache_lock = threading.Lock()
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        # Check cache first
        with self.cache_lock:
            if idx in self.cache:
                return self.cache[idx], idx, image_path
        
        try:
            # Load image
            image = Image.open(image_path).convert("RGB")
            
            if self.transform:
                image = self.transform(image)
            
            # Cache if enabled
            if len(self.cache) < 1000:  # Limit cache size
                with self.cache_lock:
                    self.cache[idx] = image
            
            return image, idx, image_path
            
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            # Return a blank image on error
            return Image.new('RGB', (224, 224), color='black'), idx, image_path


class BLIP2ImageAnalyzer:
    """Image analyzer using BLIP2 with multi-stage classification"""
    
    def __init__(self, blip2_config: dict, models_base_dir: Path, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.models_base_dir = models_base_dir
        
        # BLIP2 for classification
        self.blip2_config = blip2_config
        self.blip2_processor = None
        self.blip2_model = None
        
        # Model info
        self.model_size = blip2_config.get("model_dir", "").split("-")[-1]
        
        # Cache directory
        from config import CACHE_DIR, ENABLE_IMAGE_CACHE
        self.cache_dir = CACHE_DIR if ENABLE_IMAGE_CACHE else None
        
        # RTX 5090 detection
        self.is_rtx5090 = False
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            self.is_rtx5090 = "5090" in gpu_name
            if self.is_rtx5090:
                logger.info("RTX 5090 detected - using optimized settings")
    
    def load_models(self):
        """Load BLIP2 model"""
        logger.info("Loading BLIP2 model for analysis...")
        self._load_blip2()
        logger.info("Model loaded successfully!")
    
    def _load_blip2(self):
        """Load BLIP2 model for classification"""
        logger.info(f"Loading BLIP2-{self.model_size.upper()} model...")
        
        model_dir = self.models_base_dir / self.blip2_config["model_dir"]
        model_dir.mkdir(parents=True, exist_ok=True)
        
        processor_path = model_dir / "processor"
        model_path = model_dir / "model"
        
        try:
            if processor_path.exists() and model_path.exists():
                logger.info("Loading BLIP2 from local directory...")
                self.blip2_processor = Blip2Processor.from_pretrained(
                    processor_path, use_fast=True
                )
                self.blip2_model = Blip2ForConditionalGeneration.from_pretrained(
                    model_path,
                    torch_dtype=torch.float32
                )
            else:
                logger.info("Downloading BLIP2 model...")
                self.blip2_processor = Blip2Processor.from_pretrained(
                    self.blip2_config["name"], use_fast=True
                )
                self.blip2_model = Blip2ForConditionalGeneration.from_pretrained(
                    self.blip2_config["name"],
                    torch_dtype=torch.float32
                )
                
                # Save locally
                processor_path.mkdir(parents=True, exist_ok=True)
                model_path.mkdir(parents=True, exist_ok=True)
                self.blip2_processor.save_pretrained(processor_path)
                self.blip2_model.save_pretrained(model_path)
            
            self.blip2_model = self.blip2_model.to(self.device)
            self.blip2_model.eval()
            
        except Exception as e:
            logger.error(f"Error loading BLIP2: {e}")
            raise

    def analyze_batch_comprehensive(self, images: List[Image.Image], 
                                  image_paths: List[Path], 
                                  image_indices: List[int]) -> List[Dict[str, Any]]:
        """Comprehensive analysis with multi-stage classification"""
        start_time = time.time()
        batch_size = len(images)
        results = []
        
        try:
            # Filter valid interior images first
            valid_indices = []
            for i, path in enumerate(image_paths):
                if is_valid_interior_image(str(path)):
                    valid_indices.append(i)
                else:
                    # Add excluded result
                    results.append({
                        'ImagePath': str(path),
                        'ImageName': path.name,
                        'ImageOrder': image_indices[i],
                        'RoomType': 'excluded',
                        'Confidence': 0.0,
                        'ClassificationMethod': 'pre_filter',
                        'MultiStageResults': {'filter': 'invalid_interior_image'},
                        'BLIP2Response1': '',
                        'BLIP2Response2': '',
                        'OverallQualityScore': 0.0,
                        'ProcessingTime': 0.0
                    })
            
            if not valid_indices:
                return results
            
            # Process only valid images
            valid_images = [images[i] for i in valid_indices]
            valid_paths = [image_paths[i] for i in valid_indices]
            valid_image_indices = [image_indices[i] for i in valid_indices]
            
            # 1. Multi-stage BLIP2 Classification
            classifications = self._multi_stage_classification(valid_images, valid_paths)
            
            # 2. Technical Quality Analysis (parallel)
            with ThreadPoolExecutor(max_workers=min(len(valid_images), 8)) as executor:
                tech_futures = {
                    executor.submit(analyze_image_quality, path): idx 
                    for idx, path in enumerate(valid_paths)
                }
                
                tech_qualities = [None] * len(valid_images)
                for future in as_completed(tech_futures):
                    idx = tech_futures[future]
                    tech_qualities[idx] = future.result()
            
            # 3. Calculate consistency scores
            consistency_scores = self._calculate_batch_consistency(classifications)
            
            # 4. Combine all results
            for i in range(len(valid_images)):
                # Calculate response quality from BLIP2 responses
                blip2_responses = {
                    'BLIP2Response1': classifications[i]['all_results'][0]['response'] if len(classifications[i]['all_results']) > 0 else '',
                    'BLIP2Response2': classifications[i]['all_results'][1]['response'] if len(classifications[i]['all_results']) > 1 else '',
                }
                
                response_quality = calculate_response_quality(blip2_responses)
                
                # Calculate overall quality score
                overall_quality = calculate_overall_quality_score(
                    tech_qualities[i],
                    classifications[i]['confidence'],
                    response_quality
                )
                
                # Compile comprehensive result
                result = {
                    'ImagePath': str(valid_paths[i]),
                    'ImageName': valid_paths[i].name,
                    'ImageOrder': valid_image_indices[i],
                    
                    # Classification
                    'RoomType': classifications[i]['room_type'],
                    'Confidence': classifications[i]['confidence'],
                    'ClassificationMethod': 'multi_stage',
                    'MultiStageResults': classifications[i].get('all_results', {}),
                    'ConsistencyScore': consistency_scores[i],
                    
                    # BLIP2 Responses
                    **blip2_responses,
                    
                    # Technical quality
                    **tech_qualities[i],
                    'OverallQualityScore': overall_quality,
                    
                    # Metadata
                    'ProcessingTime': round(time.time() - start_time, 3)
                }
                
                results.append(result)
                
                # Save to cache if enabled
                if self.cache_dir:
                    cache_key = get_cache_key(str(valid_paths[i]), 'comprehensive_analysis')
                    save_to_cache(cache_key, result, self.cache_dir)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {e}")
            # Return basic results on error
            return self._fallback_analysis(images, image_paths, image_indices)

    def _multi_stage_classification(self, images: List[Image.Image], 
                               image_paths: List[Path]) -> List[Dict]:
        """Multi-stage classification using different prompts"""
        from config import CLASSIFICATION_PROMPTS
        
        results = []
        
        for i, (image, path) in enumerate(zip(images, image_paths)):
            # Check cache first
            if self.cache_dir:
                cache_key = get_cache_key(str(path), 'classification')
                cached_result = load_from_cache(cache_key, self.cache_dir)
                if cached_result:
                    results.append(cached_result)
                    continue
            
            image_results = []
            
            # Process both prompts
            for j, prompt in enumerate(CLASSIFICATION_PROMPTS):
                response = self._process_single_prompt(image, prompt)
                
                # معالجة خاصة لكل سؤال
                if j == 0:  # السؤال الأول - التصنيف المباشر
                    # تنظيف الإجابة
                    clean_response = response.strip().lower()
                    
                    # معالجة الإجابات المباشرة
                    if clean_response == 'indoor':
                        # إذا أجاب indoor فقط، نحتاج المزيد من المعلومات
                        room_type = 'unknown'
                        confidence = 0.0
                        details = "Indoor only - need description for room type"
                    elif clean_response in ['bedroom', 'kitchen', 'bathroom']:
                        room_type = clean_response
                        confidence = 0.7
                        details = "Direct room classification"
                    elif clean_response in ['living room', 'living']:
                        room_type = 'living_room'
                        confidence = 0.7
                        details = "Direct room classification"
                    elif clean_response == 'outdoor':
                        room_type = 'other'
                        confidence = 0.95
                        details = "Outdoor detected"
                    else:
                        # إذا كانت الإجابة غير واضحة، حاول استخراج المعلومات
                        room_type, confidence, details = self._parse_first_prompt_response(response)
                
                else:  # السؤال الثاني - الوصف التفصيلي
                    room_type, confidence, details = extract_room_type_from_description(response, j)
                
                image_results.append({
                    'prompt': prompt,
                    'response': response,
                    'type': room_type,
                    'confidence': confidence,
                    'details': details
                })
            
            # تحليل النتائج
            final_result = self._analyze_two_stage_results(image_results)
            
            results.append(final_result)
            
            # Save to cache
            if self.cache_dir:
                save_to_cache(cache_key, final_result, self.cache_dir)
        
        return results

    def _parse_first_prompt_response(self, response: str) -> Tuple[str, float, str]:
        """Parse response from first prompt when it's not a simple classification"""
        response_lower = response.lower()
        
        # Check for outdoor
        if 'outdoor' in response_lower or 'exterior' in response_lower:
            return 'other', 0.95, "Outdoor detected"
        
        # إذا كانت الإجابة "indoor" فقط
        if response_lower.strip() == 'indoor':
            return 'unknown', 0.0, "Indoor only - need more info"
        
        # Check for room types
        room_mapping = {
            'bedroom': ['bedroom', 'bed room'],
            'bathroom': ['bathroom', 'bath room'],
            'kitchen': ['kitchen'],
            'living_room': ['living room', 'living', 'livingroom']
        }
        
        for room_type, keywords in room_mapping.items():
            for keyword in keywords:
                if keyword in response_lower:
                    return room_type, 0.6, f"Extracted from response: {keyword}"
        
        # Check if it's clearly interior but type unclear
        if 'interior' in response_lower or 'indoor' in response_lower:
            return 'unknown', 0.0, "Interior confirmed but type unclear"
        
        return 'unknown', 0.0, "Could not parse response"

    def _analyze_two_stage_results(self, image_results: List[Dict]) -> Dict:
        """Analyze results from two-stage classification"""
        
        if len(image_results) < 2:
            return {
                'room_type': 'unknown',
                'confidence': 0.0,
                'all_results': image_results,
                'classification_method': 'insufficient_results',
                'reasoning': 'Not enough classification results'
            }
        
        first_result = image_results[0]  # التصنيف المباشر
        second_result = image_results[1]  # الوصف التفصيلي
        
        # 1. التحقق من الصور الخارجية
        if first_result['type'] == 'other' or 'outdoor' in first_result['response'].lower():
            return {
                'room_type': 'other',
                'confidence': 0.0,
                'all_results': image_results,
                'classification_method': 'outdoor_detected',
                'reasoning': 'Outdoor scene'
            }
        
        # 2. معالجة خاصة لحالة "indoor" فقط في السؤال الأول
        if first_result['response'].strip().lower() == 'indoor' and first_result['type'] == 'unknown':
            # الاعتماد بالكامل على السؤال الثاني
            second_response_lower = second_result['response'].lower()
            
            # فحص الاستبعادات أولاً
            strong_non_room_indicators = [
                'doorway', 'door way', 'entrance door', 'main door',
                'hallway', 'corridor', 'staircase', 'stairway',
                'elevator', 'lobby', 'foyer', 'threshold',
                'balcony', 'terrace', 'patio'
            ]
            
            has_strong_non_room = any(indicator in second_response_lower for indicator in strong_non_room_indicators)
            
            if has_strong_non_room:
                # تحقق من وجود عناصر غرفة حقيقية
                room_elements = [
                    'bed', 'mattress', 'pillow', 'nightstand',
                    'sofa', 'couch', 'television', 'coffee table',
                    'kitchen', 'stove', 'refrigerator', 'sink',
                    'bathroom', 'toilet', 'shower', 'bathtub'
                ]
                has_room_elements = any(elem in second_response_lower for elem in room_elements)
                
                if not has_room_elements:
                    return {
                        'room_type': 'excluded',
                        'confidence': 0.0,
                        'all_results': image_results,
                        'classification_method': 'excluded_non_room',
                        'reasoning': f'Indoor but non-room space: {second_result["response"][:50]}...'
                    }
            
            # إذا لم يكن استبعاد، استخدم تصنيف السؤال الثاني
            if second_result['type'] not in ['unknown', 'excluded', 'other']:
                return {
                    'room_type': second_result['type'],
                    'confidence': second_result['confidence'],
                    'all_results': image_results,
                    'classification_method': 'second_prompt_classification',
                    'reasoning': 'First prompt was indoor only, using description classification'
                }
        
        # 3. التحقق من التناقضات الحرجة (للحالات الأخرى)
        second_response_lower = second_result['response'].lower()
        
        # كلمات تشير بقوة إلى أنها ليست غرفة
        strong_non_room_indicators = [
            'doorway', 'door way', 'entrance door', 'main door',
            'hallway', 'corridor', 'staircase', 'stairway',
            'elevator', 'lobby', 'foyer', 'threshold',
            'balcony', 'terrace', 'patio'
        ]
        
        # فحص وجود مؤشرات قوية على أنها ليست غرفة
        has_strong_non_room = any(indicator in second_response_lower for indicator in strong_non_room_indicators)
        
        # فحص وجود عناصر غرفة حقيقية
        room_elements = {
            'bedroom': ['bed', 'mattress', 'pillow', 'blanket', 'nightstand', 'dresser', 'wardrobe'],
            'kitchen': ['kitchen', 'stove', 'oven', 'refrigerator', 'fridge', 'sink', 'counter', 'cabinet'],
            'bathroom': ['bathroom', 'toilet', 'shower', 'bathtub', 'sink', 'mirror', 'towel'],
            'living_room': ['sofa', 'couch', 'television', 'tv', 'coffee table', 'armchair']
        }
        
        # فحص التطابق بين التصنيف والوصف
        if first_result['type'] in room_elements:
            expected_elements = room_elements[first_result['type']]
            has_matching_elements = any(elem in second_response_lower for elem in expected_elements)
        else:
            has_matching_elements = False
        
        # فحص وجود أي عناصر غرفة
        all_room_elements = [elem for elements in room_elements.values() for elem in elements]
        has_any_room_elements = any(elem in second_response_lower for elem in all_room_elements)
        
        # 4. اتخاذ القرار بناءً على التحليل
        
        # حالة 1: وصف يشير بوضوح إلى مكان غير غرفة وبدون عناصر غرفة
        if has_strong_non_room and not has_any_room_elements:
            return {
                'room_type': 'excluded',
                'confidence': 0.0,
                'all_results': image_results,
                'classification_method': 'excluded_non_room',
                'reasoning': f'Non-room space: {second_result["response"][:50]}...'
            }
        
        # حالة 2: تناقض واضح - التصنيف الأول يقول غرفة لكن الوصف لا يحتوي على عناصرها
        if first_result['type'] not in ['unknown', 'excluded'] and not has_matching_elements and has_strong_non_room:
            return {
                'room_type': 'excluded',
                'confidence': 0.0,
                'all_results': image_results,
                'classification_method': 'contradiction_detected',
                'reasoning': f'First classified as {first_result["type"]} but description shows non-room space'
            }
        
        # حالة 3: التصنيف والوصف متطابقان
        if first_result['type'] == second_result['type'] and has_matching_elements:
            # ثقة عالية عند التطابق
            avg_confidence = (first_result['confidence'] + second_result['confidence']) / 2
            boosted_confidence = min(avg_confidence * 1.2, 0.95)
            
            return {
                'room_type': first_result['type'],
                'confidence': boosted_confidence,
                'all_results': image_results,
                'classification_method': 'consistent_classification',
                'reasoning': 'Classification and description match'
            }
        
        # حالة 4: الوصف يحتوي على عناصر غرفة واضحة
        if second_result['type'] not in ['unknown', 'excluded'] and second_result['type'] != 'other':
            # إعطاء الأولوية للوصف التفصيلي إذا كان واضحاً
            if 'STRONG:' in second_result['details']:
                return {
                    'room_type': second_result['type'],
                    'confidence': second_result['confidence'],
                    'all_results': image_results,
                    'classification_method': 'description_priority',
                    'reasoning': 'Strong keywords in description'
                }
        
        # حالة 5: التصنيف الأول واضح والوصف لا يناقضه
        if first_result['type'] not in ['unknown', 'excluded'] and not has_strong_non_room:
            # استخدام التصنيف الأول مع تقليل الثقة قليلاً
            return {
                'room_type': first_result['type'],
                'confidence': first_result['confidence'] * 0.8,
                'all_results': image_results,
                'classification_method': 'first_classification',
                'reasoning': 'Using initial classification with reduced confidence'
            }
        
        # حالة 6: إذا كان السؤال الثاني له تصنيف واضح
        if second_result['type'] not in ['unknown', 'excluded', 'other']:
            return {
                'room_type': second_result['type'],
                'confidence': second_result['confidence'] * 0.9,
                'all_results': image_results,
                'classification_method': 'second_classification_fallback',
                'reasoning': 'Using second prompt classification as fallback'
            }
        
        # حالة 7: غير واضح
        return {
            'room_type': 'unknown',
            'confidence': 0.0,
            'all_results': image_results,
            'classification_method': 'unclear',
            'reasoning': 'Unable to determine room type'
        }

    def _process_with_reduced_confidence(self, image_results: List[Dict], 
                                    confidence_multiplier: float, 
                                    reason: str) -> Dict:
        """Process results with reduced confidence due to conflicts"""
        final_type, base_confidence, voting_reason = self._ensemble_voting(image_results)
        
        return {
            'room_type': final_type,
            'confidence': base_confidence * confidence_multiplier,
            'all_results': image_results,
            'classification_method': 'reduced_confidence',
            'reasoning': f'{reason}. {voting_reason}'
        }

    def _enhanced_ensemble_voting(self, image_results: List[Dict]) -> Dict:
        """Enhanced voting that considers response quality"""
        weighted_votes = defaultdict(float)
        
        for i, result in enumerate(image_results):
            if result['type'] not in ['unknown', 'excluded', 'other']:
                # وزن مختلف لكل سؤال
                weight = 1.0
                if i == 0:  # التصنيف المباشر
                    weight = 0.3 if result['confidence'] < 0.7 else 0.5
                elif i == 1:  # الوصف الأول
                    weight = 1.0  # وزن أعلى للوصف التفصيلي
                elif i == 2:  # الوصف الثاني
                    weight = 0.8
                
                weighted_votes[result['type']] += result['confidence'] * weight
        
        if not weighted_votes:
            return {
                'room_type': 'unknown',
                'confidence': 0.0,
                'all_results': image_results,
                'classification_method': 'no_votes',
                'reasoning': 'No valid votes'
            }
        
        # اختيار الأفضل
        best_type = max(weighted_votes.items(), key=lambda x: x[1])
        total_weight = sum(weighted_votes.values())
        
        # حساب الثقة النهائية
        consensus = best_type[1] / total_weight if total_weight > 0 else 0
        
        # الثقة النهائية تعتمد على الإجماع وجودة الأوصاف
        type_results = [r for r in image_results if r['type'] == best_type[0]]
        avg_confidence = sum(r['confidence'] for r in type_results) / len(type_results) if type_results else 0
        
        final_confidence = (consensus * 0.6 + avg_confidence * 0.4)
        
        return {
            'room_type': best_type[0],
            'confidence': final_confidence,
            'all_results': image_results,
            'classification_method': 'weighted_ensemble_voting',
            'reasoning': f'Weighted votes: {dict(weighted_votes)}, Consensus: {consensus:.2f}'
        }
    
    def _analyze_classification_results(self, image_results: List[Dict]) -> Dict:
        """Analyze classification results with special handling for exclusions"""
        
        # التحقق من وجود استبعادات في أي من النتائج
        exclusions = [r for r in image_results if r['type'] in ['excluded', 'other']]
        valid_classifications = [r for r in image_results if r['type'] not in ['excluded', 'other', 'unknown']]
        
        # إذا كان هناك استبعاد في الوصف التفصيلي (السؤال الثالث)
        if len(image_results) >= 3 and image_results[2]['type'] == 'excluded':
            # إذا كان السؤال الأول صنفه كغرفة، فالاستبعاد له الأولوية
            if len(exclusions) >= 2 or (exclusions and not valid_classifications):
                return {
                    'room_type': 'excluded',
                    'confidence': 0.0,  # ثقة صفر للمستبعدات
                    'all_results': image_results,
                    'classification_method': 'excluded_by_description',
                    'reasoning': 'Excluded based on detailed description'
                }
        
        # إذا كان outdoor في السؤال الأول
        if image_results[0]['type'] == 'other' or 'outdoor' in image_results[0]['response'].lower():
            return {
                'room_type': 'other',
                'confidence': 0.0,  # ثقة صفر للخارجية
                'all_results': image_results,
                'classification_method': 'outdoor_detected',
                'reasoning': 'Outdoor scene detected'
            }
        
        # إذا كانت هناك تصنيفات صالحة، استخدم ensemble voting
        if valid_classifications:
            final_type, final_confidence, reasoning = self._ensemble_voting(image_results)
            
            # تحقق إضافي: إذا كان هناك تناقض قوي مع الاستبعاد
            if exclusions and final_confidence < 0.5:
                return {
                    'room_type': 'excluded',
                    'confidence': 0.0,
                    'all_results': image_results,
                    'classification_method': 'excluded_due_to_contradiction',
                    'reasoning': 'Contradictory classification with exclusion indicators'
                }
            
            return {
                'room_type': final_type,
                'confidence': final_confidence,
                'all_results': image_results,
                'classification_method': 'ensemble_voting',
                'reasoning': reasoning
            }
        
        # إذا لم تكن هناك تصنيفات صالحة
        return {
            'room_type': 'unknown',
            'confidence': 0.0,
            'all_results': image_results,
            'classification_method': 'no_valid_classification',
            'reasoning': 'No valid room type detected'
        }

    def _process_single_prompt(self, image: Image.Image, prompt: str) -> str:
        """Process a single image with a specific prompt"""
        try:
            inputs = self.blip2_processor(
                images=image,
                text=prompt,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                generated_ids = self.blip2_model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False
                )
                
            response = self.blip2_processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )[0].strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing prompt: {e}")
            return ""
    
    def _ensemble_voting(self, results: List[Dict]) -> Tuple[str, float, str]:
        """Perform ensemble voting with keyword priority for contradictions"""
        valid_results = [
            r for r in results 
            if r['type'] not in ['unknown', 'excluded'] and r['confidence'] > 0
        ]
        
        if not valid_results:
            return 'unknown', 0.0, "No valid classifications"
        
        # تحليل الكلمات المفتاحية في جميع الاستجابات
        keyword_based_types = defaultdict(float)
        response_based_types = defaultdict(float)
        
        for result in valid_results:
            # فحص وجود كلمات مفتاحية قوية في الاستجابة
            response_text = result.get('response', '').lower()
            details = result.get('details', '')
            
            # إذا كانت هناك كلمات مفتاحية قوية، أعط وزن أكبر
            if 'STRONG:' in details:
                keyword_based_types[result['type']] += result['confidence'] * 2.0
            elif 'MEDIUM:' in details:
                keyword_based_types[result['type']] += result['confidence'] * 1.5
            else:
                response_based_types[result['type']] += result['confidence']
        
        # دمج النتائج مع إعطاء الأولوية للكلمات المفتاحية
        combined_votes = defaultdict(float)
        
        # إضافة أصوات الكلمات المفتاحية (وزن أعلى)
        for room_type, score in keyword_based_types.items():
            combined_votes[room_type] += score * 1.5
        
        # إضافة أصوات التصنيف العادي
        for room_type, score in response_based_types.items():
            combined_votes[room_type] += score
        
        # في حالة التناقض، تحقق من الكلمات المفتاحية في جميع الاستجابات
        if len(combined_votes) > 1:
            # جمع كل الاستجابات
            all_responses = ' '.join([r.get('response', '') for r in results]).lower()
            
            # فحص مباشر للكلمات المفتاحية القوية
            strong_room_type = self._detect_strong_keywords(all_responses)
            if strong_room_type:
                # إذا وُجدت كلمات مفتاحية قوية، أعطها الأولوية
                combined_votes[strong_room_type] *= 2.0
        
        # Get best result
        if not combined_votes:
            return 'unknown', 0.0, "No valid votes"
        
        best_type = max(combined_votes.items(), key=lambda x: x[1])
        final_type = best_type[0]
        
        # حساب الثقة النهائية
        type_results = [r for r in valid_results if r['type'] == final_type]
        if type_results:
            # إذا كان هناك كلمات مفتاحية قوية، زيادة الثقة
            has_strong_keywords = any('STRONG:' in r.get('details', '') for r in type_results)
            base_confidence = sum(r['confidence'] for r in type_results) / len(type_results)
            
            if has_strong_keywords:
                final_confidence = min(base_confidence * 1.2, 0.95)
            else:
                final_confidence = base_confidence
        else:
            final_confidence = 0.5
        
        # Calculate consensus
        total_votes = sum(combined_votes.values())
        consensus = best_type[1] / total_votes if total_votes > 0 else 0
        
        # تحديد السبب
        if keyword_based_types.get(final_type, 0) > response_based_types.get(final_type, 0):
            reasoning = f"Strong keywords detected. Votes: {dict(combined_votes)}, Consensus: {consensus:.2f}"
        else:
            reasoning = f"Votes: {dict(combined_votes)}, Consensus: {consensus:.2f}"
        
        return final_type, final_confidence, reasoning

    def _detect_strong_keywords(self, text: str) -> Optional[str]:
        """Detect strong keywords in text and return room type"""
        text_lower = text.lower()
        
        # كلمات مفتاحية قوية جداً
        strong_indicators = {
            'kitchen': ['kitchen', 'stove', 'oven', 'refrigerator', 'fridge'],
            'bathroom': ['bathroom', 'toilet', 'shower', 'bathtub'],
            'bedroom': ['bedroom', 'bed ', ' bed', 'mattress'],
            'living_room': ['living room', 'couch', 'sofa', 'television']
        }
        
        for room_type, keywords in strong_indicators.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return room_type
        
        return None

    def _calculate_batch_consistency(self, classifications: List[Dict]) -> List[float]:
        """Calculate consistency scores for a batch of classifications"""
        consistency_scores = []
        
        # Count room types
        room_type_counts = defaultdict(int)
        for cls in classifications:
            if cls['room_type'] not in ['unknown', 'excluded', 'other']:
                room_type_counts[cls['room_type']] += 1
        
        total_valid = sum(room_type_counts.values())
        
        # Calculate consistency for each image
        for cls in classifications:
            room_type = cls['room_type']
            if room_type in room_type_counts and total_valid > 0:
                consistency = room_type_counts[room_type] / total_valid
            else:
                consistency = 0.0
            consistency_scores.append(consistency)
        
        return consistency_scores
    
    def _parse_room_type(self, answer: str) -> Optional[str]:
        """Parse room type from model answer"""
        from config import TARGET_ROOMS
        
        answer_lower = answer.lower()
        
        # Direct match
        for room in TARGET_ROOMS:
            if room.replace("_", " ") in answer_lower:
                return room
        
        # Common variations
        if "living" in answer_lower:
            return "living_room"
        elif "bed" in answer_lower:
            return "bedroom"
        elif "bath" in answer_lower:
            return "bathroom"
        elif "kitchen" in answer_lower:
            return "kitchen"
        
        return None
    
    def _fallback_analysis(self, images: List[Image.Image], 
                          image_paths: List[Path], 
                          image_indices: List[int]) -> List[Dict]:
        """Simplified analysis as fallback"""
        results = []
        
        for i, (image, path, idx) in enumerate(zip(images, image_paths, image_indices)):
            results.append({
                'ImagePath': str(path),
                'ImageName': path.name,
                'ImageOrder': idx,
                'RoomType': 'unknown',
                'Confidence': 0.0,
                'ClassificationMethod': 'fallback',
                'BLIP2Response1': '',
                'BLIP2Response2': '',
                'OverallQualityScore': 50.0,
                'ProcessingTime': 0.0
            })
        
        return results
