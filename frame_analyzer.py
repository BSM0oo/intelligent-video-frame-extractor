import cv2
import numpy as np
import pytesseract
from PIL import Image
import logging
from typing import Tuple, Optional

class FrameAnalyzer:
    """
    Handles frame analysis, comparison, and feature extraction
    """
    
    def __init__(self, 
                 scene_threshold: float = 0.7,
                 text_threshold: float = 0.8):
        self.scene_threshold = scene_threshold
        self.text_threshold = text_threshold
        self.logger = logging.getLogger(__name__)
        
    def calculate_histogram_difference(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """
        Calculate difference between color histograms of two frames
        """
        hist1 = cv2.calcHist([frame1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist2 = cv2.calcHist([frame2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        
        hist1 = cv2.normalize(hist1, hist1).flatten()
        hist2 = cv2.normalize(hist2, hist2).flatten()
        
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    
    def detect_edges(self, frame: np.ndarray) -> float:
        """
        Detect edges in frame and return edge density
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        return np.mean(edges) / 255.0
    
    def assess_frame_quality(self, frame: np.ndarray) -> float:
        """
        Assess frame quality (blur detection)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    
    def compute_frame_hash(self, frame: np.ndarray) -> str:
        """
        Compute a perceptual hash of the frame for quick comparison
        """
        small_frame = cv2.resize(frame, (8, 8))
        gray_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        return hashlib.md5(gray_frame.tobytes()).hexdigest()
    
    def extract_text(self, frame: np.ndarray) -> str:
        """
        Extract text from a frame using OCR
        """
        try:
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            return pytesseract.image_to_string(pil_image)
        except Exception as e:
            self.logger.error(f"Error extracting text: {str(e)}")
            return ""
    
    def compare_frames(self, 
                      current_frame: np.ndarray, 
                      prev_frame: np.ndarray,
                      prev_text: str) -> Tuple[bool, str]:
        """
        Compare two frames using multiple metrics
        Returns: (is_significant_change, new_text)
        """
        # Calculate histogram difference
        hist_diff = self.calculate_histogram_difference(current_frame, prev_frame)
        
        # Skip further processing if frames are too similar
        if hist_diff > 0.95:
            return False, prev_text
        
        # Extract and compare text
        current_text = self.extract_text(current_frame)
        
        # Calculate visual difference
        frame_diff = cv2.absdiff(current_frame, prev_frame)
        frame_change = np.mean(frame_diff) / 255.0
        
        # Calculate text difference
        from difflib import SequenceMatcher
        text_ratio = SequenceMatcher(None, prev_text, current_text).ratio()
        
        # Calculate edge difference
        edge_diff = abs(self.detect_edges(current_frame) - self.detect_edges(prev_frame))
        
        # Assess frame quality
        quality = self.assess_frame_quality(current_frame)
        
        # Combine metrics for final decision
        is_scene_change = frame_change > self.scene_threshold or edge_diff > 0.3
        is_text_change = text_ratio < self.text_threshold
        is_quality_frame = quality > 100  # Arbitrary threshold for demonstration
        
        return (is_scene_change or is_text_change) and is_quality_frame, current_text