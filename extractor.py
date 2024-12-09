import cv2
import logging
from typing import List, Dict, Any, Optional
import os
from youtube_dl import YoutubeDL
import utils
from frame_analyzer import FrameAnalyzer
from llm_analyzer import LLMAnalyzer

class VideoFrameExtractor:
    """
    Main class for extracting and analyzing key frames from videos
    """
    
    def __init__(self, 
                 scene_threshold: float = 0.7,
                 text_threshold: float = 0.8,
                 min_frame_gap: int = 15,
                 output_dir: str = "extracted_frames",
                 openai_api_key: Optional[str] = None):
        """
        Initialize the frame extractor
        
        Args:
            scene_threshold: Threshold for scene change detection (0-1)
            text_threshold: Threshold for text change detection (0-1)
            min_frame_gap: Minimum number of frames between extractions
            output_dir: Directory to save extracted frames
            openai_api_key: OpenAI API key for LLM analysis
        """
        self.output_dir = output_dir
        self.min_frame_gap = min_frame_gap
        self.frame_cache = {}
        self.extracted_frames = []
        self.pending_extractions = set()
        
        # Create output directories
        self.dirs = utils.create_output_dirs(output_dir)
        
        # Initialize components
        self.frame_analyzer = FrameAnalyzer(
            scene_threshold=scene_threshold,
            text_threshold=text_threshold
        )
        self.llm_analyzer = LLMAnalyzer(api_key=openai_api_key)
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
    
    def download_youtube_video(self, url: str) -> str:
        """
        Download a YouTube video and return the path to the downloaded file
        """
        ydl_opts = {
            'format': 'best[ext=mp4]',
            'outtmpl': os.path.join(self.dirs['frames'], 'video.%(ext)s')
        }
        
        with YoutubeDL(ydl_opts) as ydl:
            self.logger.info(f"Downloading video from {url}")
            ydl.download([url])
            return os.path.join(self.dirs['frames'], 'video.mp4')
    
    def process_video(self,
                     video_path: str,
                     start_time: Optional[float] = None,
                     end_time: Optional[float] = None,
                     analyze_interval: Optional[int] = 50):
        """
        Process the video and extract key frames
        
        Args:
            video_path: Path to the video file
            start_time: Optional start time in seconds
            end_time: Optional end time in seconds
            analyze_interval: Number of frames between LLM analyses
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Error opening video file: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Set start and end frames
        start_frame = int(start_time * fps) if start_time else 0
        end_frame = int(end_time * fps) if end_time else total_frames
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        prev_frame = None
        prev_text = ""
        frame_count = start_frame
        last_saved_frame = -self.min_frame_gap
        
        self.logger.info(f"Processing video: {video_path}")
        self.logger.info(f"Total frames to process: {end_frame - start_frame}")
        
        while frame_count < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
                
            if prev_frame is None:
                prev_frame = frame
                continue
            
            # Check for significant changes
            if frame_count - last_saved_frame >= self.min_frame_gap:
                is_change, new_text = self.frame_analyzer.compare_frames(
                    frame, prev_frame, prev_text
                )
                
                if is_change:
                    # Save frame
                    timestamp = frame_count / fps
                    frame_path = os.path.join(
                        self.dirs['frames'],
                        f"frame_{timestamp:.2f}.jpg"
                    )
                    cv2.imwrite(frame_path, frame)
                    
                    self.extracted_frames.append({
                        'timestamp': timestamp,
                        'path': frame_path,
                        'text': new_text,
                        'frame_number': frame_count
                    })
                    
                    self.logger.info(f"Saved frame at timestamp {timestamp:.2f}s")
                    last_saved_frame = frame_count
                    prev_text = new_text
            
            prev_frame = frame
            frame_count += 1
            
            # Log progress
            if frame_count % 1000 == 0:
                progress = (frame_count - start_frame) / (end_frame - start_frame) * 100
                self.logger.info(f"Progress: {progress:.1f}%")
            
            # Periodic LLM analysis
            if analyze_interval and len(self.extracted_frames) >= analyze_interval:
                last_n_frames = self.extracted_frames[-analyze_interval:]
                frame_analyses = []
                
                for frame_info in last_n_frames:
                    analysis = self.llm_analyzer.analyze_frame(
                        frame_info['path'], frame_info
                    )
                    if analysis:
                        frame_analyses.append(analysis)
                
                if frame_analyses:
                    sequence_analysis = self.llm_analyzer.analyze_sequence(frame_analyses)
                    missing_timepoints = self.llm_analyzer.find_missing_timepoints(sequence_analysis)
                    
                    if missing_timepoints:
                        self.logger.info(f"LLM identified potential missing frames at: {missing_timepoints}")
                        self.pending_extractions.update(missing_timepoints)
        
        cap.release()
        
        # Save metadata
        utils.save_metadata(self.extracted_frames, self.output_dir, self.logger)
        
        # Process any missing frames identified during extraction
        if self.pending_extractions:
            self.logger.info(f"Processing {len(self.pending_extractions)} identified missing frames...")
            self.review_and_extract_missing(video_path, list(self.pending_extractions))
    
    def review_and_extract_missing(self,
                                 video_path: str,
                                 time_points: Optional[List[float]] = None,
                                 window_size: float = 2.0):
        """
        Review extracted frames and extract any missing content
        
        Args:
            video_path: Path to the video file
            time_points: Optional list of specific timestamps to analyze
            window_size: Time window around each point to analyze (in seconds)
        """
        # If no specific timepoints provided, analyze all frames
        if not time_points:
            frame_analyses = []
            for frame_info in self.extracted_frames:
                analysis = self.llm_analyzer.analyze_frame(
                    frame_info['path'], frame_info
                )
                if analysis:
                    frame_analyses.append(analysis)
            
            if frame_analyses:
                sequence_analysis = self.llm_analyzer.analyze_sequence(frame_analyses)
                time_points = self.llm_analyzer.find_missing_timepoints(sequence_analysis)
        
        if not time_points:
            self.logger.info("No additional frames needed based on analysis")
            return
        
        # Extract additional frames
        for timestamp in time_points:
            self.logger.info(f"Extracting additional frames around {timestamp}s")
            start_time = max(0, timestamp - window_size/2)
            end_time = timestamp + window_size/2
            self.process_video(video_path, start_time, end_time)
        
        # Save updated metadata
        utils.save_metadata(self.extracted_frames, self.output_dir, self.logger)
    
    def generate_pdf(self, output_path: str = "extracted_frames.pdf"):
        """
        Generate a PDF from extracted frames
        """
        utils.generate_pdf(self.extracted_frames, output_path, self.logger)