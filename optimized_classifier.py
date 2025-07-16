"""
Main image classification and analysis processor optimized_classifier.py
Using BLIP2 for multi-stage classification with comprehensive analysis
"""
import os
import sys
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from collections import defaultdict
import pandas as pd
import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
import torch
from PIL import Image
import psutil
import GPUtil
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from config import *
from utils import *
from batch_processor import BLIP2ImageAnalyzer, PropertyImageDataset

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

console = Console()


@dataclass
class ProcessingStats:
    """Statistics tracking for processing"""
    total_images_scanned: int = 0
    total_images_analyzed: int = 0
    total_images_classified: int = 0
    total_images_excluded: int = 0
    total_outdoor_detected: int = 0
    properties_processed: int = 0
    properties_completed: int = 0
    start_time: Optional[float] = None
    current_property: Optional[str] = None
    gpu_memory_used: float = 0.0
    cpu_percent: float = 0.0
    images_per_second: float = 0.0
    classification_methods: Dict[str, int] = None
    
    def __post_init__(self):
        if self.classification_methods is None:
            self.classification_methods = defaultdict(int)


class ComprehensiveImageProcessor:
    """Main processor for comprehensive image analysis with multi-stage classification"""
    
    def __init__(self, model_size: str, max_properties: Optional[int] = None, 
                 max_images: Optional[int] = None, custom_batch_size: Optional[int] = None):
        self.model_size = model_size
        self.max_properties = max_properties
        self.max_images = max_images
        
        # Get model configuration
        if model_size not in AVAILABLE_MODELS:
            raise ValueError(f"Model size must be one of: {list(AVAILABLE_MODELS.keys())}")
        
        self.model_config = AVAILABLE_MODELS[model_size]
        self.batch_size = custom_batch_size or self.model_config["batch_size"]
        
        # Initialize analyzer
        self.analyzer = BLIP2ImageAnalyzer(
            self.model_config, 
            MODELS_BASE_DIR, 
            DEVICE
        )
        
        self.processed_properties: Set[str] = set()
        self.stats = ProcessingStats()
        
    def initialize(self):
        """Initialize the processor"""
        console.print(f"[bold green]Initializing Comprehensive Image Analysis System[/bold green]")
        console.print(f"[cyan]Model: BLIP2-{self.model_size.upper()}[/cyan]")
        console.print(f"[cyan]Classification: Multi-Stage Approach[/cyan]")
        
        # System info
        self._print_system_info()
        
        # Model configuration
        model_info = Panel(
            f"""[cyan]BLIP2 Model:[/cyan] {self.model_config['name']}
[cyan]Batch Size:[/cyan] {self.batch_size}
[cyan]Analysis Mode:[/cyan] Comprehensive with Multi-Stage Classification
[cyan]Classification Stages:[/cyan] {len(CLASSIFICATION_PROMPTS)}
[cyan]Output:[/cyan] {OUTPUT_COMPREHENSIVE_FILE}""",
            title="Configuration",
            border_style="blue"
        )
        console.print(model_info)
        
        # Ensure directories
        ensure_directories()
        
        # Load models
        self.analyzer.load_models()
        
        # Load existing analysis
        self.existing_analysis = load_existing_analysis(OUTPUT_COMPREHENSIVE_FILE)
        self.processed_properties = set(self.existing_analysis.keys())
        
        # Get last processed property
        self.last_property_id = get_last_processed_property(OUTPUT_COMPREHENSIVE_FILE)
        
        console.print(f"[green]✓ Initialization complete![/green]")
        console.print(f"[cyan]Found {len(self.processed_properties)} already processed properties[/cyan]")
    
    def _print_system_info(self):
        """Print system information"""
        gpus = GPUtil.getGPUs()
        
        info_table = Table(title="System Information", expand=True)
        info_table.add_column("Component", style="cyan")
        info_table.add_column("Details", style="green")
        
        # CPU info
        info_table.add_row("CPU", f"{psutil.cpu_count()} cores, {psutil.cpu_freq().current:.0f} MHz")
        info_table.add_row("RAM", f"{psutil.virtual_memory().total / (1024**3):.1f} GB")
        
        # GPU info
        if gpus:
            gpu = gpus[0]
            info_table.add_row("GPU", f"{gpu.name}")
            info_table.add_row("GPU Memory", f"{gpu.memoryTotal} MB")
            info_table.add_row("GPU Driver", f"{gpu.driver}")
        
        # PyTorch info
        info_table.add_row("PyTorch", torch.__version__)
        info_table.add_row("CUDA", torch.version.cuda if torch.cuda.is_available() else "N/A")
        
        console.print(info_table)
    
    def process_property_comprehensive(self, property_id: str, 
                                     images_df: pd.DataFrame) -> Dict[str, Any]:
        """Process all images for a property with comprehensive analysis (محسنة للأداء)"""
        import time
        property_results = {
            'property_id': property_id,
            'images_analyzed': [],
            'best_images': {},
            'processing_time': 0,
            'classification_stats': defaultdict(int)
        }
        start_time = time.time()
        images_list = images_df.to_dict('records')
        # Limit images if specified
        if self.max_images:
            remaining = self.max_images - self.stats.total_images_scanned
            if remaining <= 0:
                return property_results
            images_list = images_list[:remaining]
        # --- تحسين: تحميل وتحقيق متوازي ---
        batch_image_paths = []
        batch_image_names = []
        batch_indices = []
        for idx, img_info in enumerate(images_list):
            image_name = img_info['ImageName']
            image_path = IMAGES_ROOT / image_name
            batch_image_paths.append(image_path)
            batch_image_names.append(image_name)
            batch_indices.append(idx)
        # تحقق متوازي للصور
        t0 = time.time()
        valid_mask = batch_is_valid_interior_images([str(p) for p in batch_image_paths])
        t1 = time.time()
        # تحميل الصور الصالحة فقط
        batch_images = []
        batch_paths = []
        batch_indices_valid = []
        for i, is_valid in enumerate(valid_mask):
            if is_valid and batch_image_paths[i].exists():
                try:
                    image = Image.open(batch_image_paths[i]).convert('RGB')
                    batch_images.append(image)
                    batch_paths.append(batch_image_paths[i])
                    batch_indices_valid.append(batch_indices[i])
                except Exception as e:
                    logger.error(f"Error loading image {batch_image_paths[i]}: {e}")
            else:
                self.stats.total_images_excluded += 1
        self.stats.total_images_scanned += len(batch_image_paths)
        t2 = time.time()
        if not batch_images:
            return property_results
        # --- نهاية التحميل والتحقق ---
        # Comprehensive analysis with multi-stage classification
        try:
            t3 = time.time()
            analysis_results = self.analyzer.analyze_batch_comprehensive(
                batch_images,
                batch_paths,
                batch_indices_valid
            )
            t4 = time.time()
            # Process results
            for result in analysis_results:
                result['PropertyID'] = property_id
                result['Timestamp'] = datetime.now().isoformat()
                # Update statistics
                room_type = result.get('RoomType', 'unknown')
                property_results['classification_stats'][room_type] += 1
                if room_type == 'other':
                    self.stats.total_outdoor_detected += 1
                elif room_type == 'excluded':
                    self.stats.total_images_excluded += 1
                # Track classification method
                method = result.get('ClassificationMethod', 'unknown')
                self.stats.classification_methods[method] += 1
                # Save to file
                save_comprehensive_analysis(OUTPUT_COMPREHENSIVE_FILE, result)
                # Add to property results
                property_results['images_analyzed'].append(result)
                self.stats.total_images_analyzed += 1
                if room_type and room_type not in ['error', 'unknown', 'excluded', 'other']:
                    self.stats.total_images_classified += 1
            # تتبع زمني
            logger.info(f"[Timing] Property {property_id}: check+load={t2-t0:.2f}s, analysis={t4-t3:.2f}s, total={t4-t0:.2f}s")
        except Exception as e:
            logger.error(f"Error processing batch for property {property_id}: {e}")
        # Select best images for this property
        if property_results['images_analyzed']:
            property_results['best_images'] = select_best_images_per_property(
                property_results['images_analyzed'],
                SELECTION_WEIGHTS
            )
            # Save best images selection
            save_best_images(
                OUTPUT_BEST_IMAGES_FILE,
                property_id,
                property_results['best_images']
            )
        property_results['processing_time'] = time.time() - start_time
        return property_results
    
    def update_system_stats(self):
        """Update system performance statistics"""
        # GPU stats
        if torch.cuda.is_available():
            self.stats.gpu_memory_used = torch.cuda.memory_allocated() / (1024**3)
        
        # CPU stats
        self.stats.cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Processing speed
        if self.stats.start_time and self.stats.total_images_scanned > 0:
            elapsed = time.time() - self.stats.start_time
            self.stats.images_per_second = self.stats.total_images_scanned / elapsed
    
    def create_status_display(self) -> Table:
        """Create status display with comprehensive metrics"""
        self.update_system_stats()
        
        table = Table(title="Comprehensive Analysis Status", expand=True)
        
        elapsed = time.time() - self.stats.start_time if self.stats.start_time else 0
        
        # Add columns
        table.add_column("Metric", style="cyan", width=35)
        table.add_column("Value", style="green", width=25)
        table.add_column("System Metric", style="cyan", width=30)
        table.add_column("Value", style="yellow", width=25)
        
        # Rows
        table.add_row(
            "Current Property", str(self.stats.current_property or 'N/A'),
            "GPU Memory", f"{self.stats.gpu_memory_used:.1f} GB"
        )
        table.add_row(
            "Properties Processed", f"{self.stats.properties_processed:,}",
            "CPU Usage", f"{self.stats.cpu_percent:.1f}%"
        )
        table.add_row(
            "Properties Completed", f"{self.stats.properties_completed:,}",
            "Processing Speed", f"{self.stats.images_per_second:.1f} img/sec"
        )
        table.add_row(
            "Images Scanned", f"{self.stats.total_images_scanned:,}",
            "GPU Utilization", f"{GPUtil.getGPUs()[0].load * 100:.1f}%" if GPUtil.getGPUs() else "N/A"
        )
        table.add_row(
            "Images Analyzed", f"{self.stats.total_images_analyzed:,}",
            "Batch Size", str(self.batch_size)
        )
        table.add_row(
            "Images Classified", f"{self.stats.total_images_classified:,}",
            "Model", f"BLIP2-{self.model_size.upper()}"
        )
        table.add_row(
            "Images Excluded", f"{self.stats.total_images_excluded:,}",
            "Outdoor Detected", f"{self.stats.total_outdoor_detected:,}"
        )
        table.add_row(
            "Classification Rate", f"{(self.stats.total_images_classified / max(self.stats.total_images_analyzed, 1) * 100):.1f}%",
            "Elapsed Time", format_time(elapsed)
        )
        
        return table
    
    def create_classification_stats_table(self) -> Table:
        """Create table showing classification method statistics"""
        table = Table(title="Classification Method Statistics", expand=False)
        table.add_column("Method", style="cyan")
        table.add_column("Count", style="green")
        table.add_column("Percentage", style="yellow")
        
        total = sum(self.stats.classification_methods.values())
        for method, count in sorted(self.stats.classification_methods.items(), 
                                   key=lambda x: x[1], reverse=True):
            percentage = (count / total * 100) if total > 0 else 0
            table.add_row(method, f"{count:,}", f"{percentage:.1f}%")
        
        return table
    
    def run(self):
        """Run the comprehensive analysis process"""
        self.initialize()
        self.stats.start_time = time.time()
        
        # Load property index
        console.print("[cyan]Loading property index...[/cyan]")
        df_index = load_property_index(INPUT_INDEX_FILE)
        df_index['PropertyID'] = df_index['PropertyID'].astype(str)
        
        # Group by PropertyID
        grouped = df_index.groupby('PropertyID', sort=False)
        property_groups = list(grouped)
        
        # Filter to start from last processed
        if self.last_property_id:
            start_idx = 0
            for i, (pid, _) in enumerate(property_groups):
                if str(pid) == str(self.last_property_id):
                    start_idx = i + 1
                    break
            property_groups = property_groups[start_idx:]
            console.print(f"[cyan]Resuming from property index {start_idx}[/cyan]")
        
        # Apply limits
        if self.max_properties:
            property_groups = property_groups[:self.max_properties]
        
        total_properties = len(property_groups)
        console.print(f"[green]Processing {total_properties} properties with multi-stage classification...[/green]")
        
        # Progress tracking
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console,
            refresh_per_second=2
        ) as progress:
            
            task = progress.add_task(
                f"[cyan]Analyzing properties comprehensively...", 
                total=total_properties
            )
            
            for prop_idx, (property_id, group) in enumerate(property_groups):
                property_id = str(property_id)
                self.stats.current_property = property_id
                self.stats.properties_processed += 1
                
                # Skip if already fully processed
                if property_id in self.existing_analysis:
                    existing_images = len(self.existing_analysis[property_id])
                    total_images = len(group)
                    if existing_images >= total_images:
                        progress.update(task, advance=1)
                        continue
                
                # Process property
                try:
                    results = self.process_property_comprehensive(property_id, group)
                    
                    if results['best_images']:
                        self.stats.properties_completed += 1
                    
                except Exception as e:
                    logger.error(f"Error processing property {property_id}: {e}")
                
                # Update progress
                progress.update(task, advance=1)
                
                # Display status periodically
                if prop_idx % 10 == 0 or prop_idx == total_properties - 1:
                    status_table = self.create_status_display()
                    console.print(status_table)
                    
                    # Show classification method stats
                    if self.stats.classification_methods:
                        class_stats_table = self.create_classification_stats_table()
                        console.print(class_stats_table)
                
                # Check image limit
                if self.max_images and self.stats.total_images_scanned >= self.max_images:
                    console.print(f"[yellow]Reached maximum image limit ({self.max_images})[/yellow]")
                    break
        
        # Final summary
        self.print_summary()
    
    def print_summary(self):
        """Print final summary"""
        elapsed = time.time() - self.stats.start_time
        
        avg_speed = self.stats.total_images_scanned / elapsed if elapsed > 0 else 0
        classification_rate = (self.stats.total_images_classified / max(self.stats.total_images_analyzed, 1) * 100)
        exclusion_rate = (self.stats.total_images_excluded / max(self.stats.total_images_scanned, 1) * 100)
        
        summary_text = f"""[bold green]Comprehensive Analysis Complete![/bold green]

Processing Statistics:
├─ Properties Processed: {self.stats.properties_processed:,}
├─ Properties Completed: {self.stats.properties_completed:,}
├─ Images Scanned: {self.stats.total_images_scanned:,}
├─ Images Analyzed: {self.stats.total_images_analyzed:,}
├─ Images Classified: {self.stats.total_images_classified:,}
├─ Images Excluded: {self.stats.total_images_excluded:,}
├─ Outdoor Detected: {self.stats.total_outdoor_detected:,}
├─ Classification Success: {classification_rate:.1f}%
└─ Exclusion Rate: {exclusion_rate:.1f}%

Performance Metrics:
├─ Total Time: {format_time(elapsed)}
├─ Average Speed: {avg_speed:.1f} images/sec
├─ GPU Memory Peak: {self.stats.gpu_memory_used:.1f} GB
└─ Analysis Mode: Multi-Stage Classification with BLIP2

Classification Methods Used:
"""
        
        # Add classification method breakdown
        total_methods = sum(self.stats.classification_methods.values())
        for method, count in sorted(self.stats.classification_methods.items(), 
                                   key=lambda x: x[1], reverse=True):
            percentage = (count / total_methods * 100) if total_methods > 0 else 0
            summary_text += f"├─ {method}: {count:,} ({percentage:.1f}%)\n"
        
        summary_text += f"""
Output Files:
├─ Comprehensive Analysis: {OUTPUT_COMPREHENSIVE_FILE}
├─ Best Images Selection: {OUTPUT_BEST_IMAGES_FILE}
└─ Log File: {LOG_FILE}"""
        
        summary = Panel(summary_text, title="Summary", border_style="green")
        console.print(summary)
        
        # Show final classification stats table
        if self.stats.classification_methods:
            console.print("\n")
            class_stats_table = self.create_classification_stats_table()
            console.print(class_stats_table)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='RE-FusionX Comprehensive Image Analysis System with Multi-Stage Classification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze with XL model (faster)
  python optimized_classifier.py --model xl --batch-size 32
  
  # Analyze with XXL model (more accurate)
  python optimized_classifier.py --model xxl --batch-size 12
  
  # Process first 100 properties
  python optimized_classifier.py --model xl --max-properties 100
  
  # Process with custom batch size
  python optimized_classifier.py --model xl --batch-size 24
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        choices=['xl', 'xxl'],
        default=DEFAULT_MODEL,
        help=f'Model size to use (default: {DEFAULT_MODEL})'
    )
    parser.add_argument(
        '--max-properties', 
        type=int, 
        help='Maximum number of properties to process'
    )
    parser.add_argument(
        '--max-images', 
        type=int, 
        help='Maximum number of images to scan'
    )
    parser.add_argument(
        '--batch-size', 
        type=int,
        help='Batch size for processing (overrides model default)'
    )
    
    args = parser.parse_args()
    
    # Print configuration
    console.print(f"\n[bold cyan]RE-FusionX Comprehensive Image Analysis[/bold cyan]")
    console.print(f"[cyan]Multi-Stage Classification with BLIP2-{args.model.upper()}[/cyan]\n")
    
    try:
        processor = ComprehensiveImageProcessor(
            model_size=args.model,
            max_properties=args.max_properties,
            max_images=args.max_images,
            custom_batch_size=args.batch_size
        )
        processor.run()
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Process interrupted by user[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        logger.exception("Fatal error occurred")
        sys.exit(1)


if __name__ == "__main__":
    main()
