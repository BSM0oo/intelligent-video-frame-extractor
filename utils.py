import logging
import os
from typing import List, Dict, Any, Optional
from PIL import Image
import json

def setup_logging(log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up logging configuration
    """
    logger = logging.getLogger('video_frame_extractor')
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def create_output_dirs(base_dir: str) -> Dict[str, str]:
    """
    Create necessary output directories
    """
    dirs = {
        'frames': os.path.join(base_dir, 'frames'),
        'analysis': os.path.join(base_dir, 'analysis'),
        'exports': os.path.join(base_dir, 'exports')
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs

def generate_pdf(frames: List[Dict[str, Any]], output_path: str, logger: logging.Logger):
    """
    Generate PDF from extracted frames
    """
    if not frames:
        logger.warning("No frames to generate PDF from")
        return
        
    images = []
    for frame_info in frames:
        try:
            img = Image.open(frame_info['path'])
            images.append(img)
        except Exception as e:
            logger.error(f"Error loading image {frame_info['path']}: {str(e)}")
            
    if images:
        images[0].save(
            output_path, 
            save_all=True, 
            append_images=images[1:],
            quality=95
        )
        logger.info(f"PDF generated: {output_path}")

def save_metadata(frames: List[Dict[str, Any]], output_dir: str, logger: logging.Logger):
    """
    Save metadata about extracted frames
    """
    try:
        metadata = {
            'total_frames': len(frames),
            'frames': frames
        }
        
        metadata_path = os.path.join(output_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Metadata saved to: {metadata_path}")
    except Exception as e:
        logger.error(f"Error saving metadata: {str(e)}")

def load_metadata(metadata_path: str) -> Dict[str, Any]:
    """
    Load metadata from JSON file
    """
    try:
        with open(metadata_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading metadata: {str(e)}")
        return {}

def print_summary(metadata: Dict[str, Any], analysis_path: Optional[str] = None):
    """
    Print summary of extraction and analysis
    """
    print("\nExtraction Summary:")
    print(f"- Total frames extracted: {metadata.get('total_frames', 0)}")
    
    if analysis_path and os.path.exists(analysis_path):
        try:
            with open(analysis_path, 'r') as f:
                analysis = json.load(f)
                print("\nLLM Analysis Summary:")
                if 'initial_analysis' in analysis:
                    print("Initial Analysis:", 
                          analysis['initial_analysis'].get('sequence_analysis', '')[:200] + '...')
                if 'final_analysis' in analysis:
                    print("Final Analysis:", 
                          analysis['final_analysis'].get('sequence_analysis', '')[:200] + '...')
        except Exception as e:
            logging.error(f"Error loading analysis: {str(e)}")
