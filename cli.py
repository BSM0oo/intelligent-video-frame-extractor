import argparse
import os
import sys
from typing import Dict, Any
from extractor import VideoFrameExtractor
import utils

def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(
        description='Extract key frames from videos with LLM analysis',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--input', '-i', required=True,
                      help='YouTube URL or local video path')
    
    # Optional arguments
    parser.add_argument('--output-dir', '-o', default='extracted_frames',
                      help='Output directory')
    parser.add_argument('--openai-key',
                      default=os.getenv('OPENAI_API_KEY'),
                      help='OpenAI API key')
    parser.add_argument('--analyze-interval', type=int, default=50,
                      help='Number of frames between LLM analyses')
    parser.add_argument('--scene-threshold', type=float, default=0.7,
                      help='Threshold for scene change detection')
    parser.add_argument('--text-threshold', type=float, default=0.8,
                      help='Threshold for text change detection')
    parser.add_argument('--min-frame-gap', type=int, default=15,
                      help='Minimum frames between extractions')
    parser.add_argument('--start-time', type=float,
                      help='Start time in seconds')
    parser.add_argument('--end-time', type=float,
                      help='End time in seconds')
    parser.add_argument('--log-file',
                      help='Path to log file')
    
    return parser.parse_args()

def validate_args(args: argparse.Namespace) -> None:
    """
    Validate command line arguments
    """
    if args.scene_threshold < 0 or args.scene_threshold > 1:
        raise ValueError("Scene threshold must be between 0 and 1")
    
    if args.text_threshold < 0 or args.text_threshold > 1:
        raise ValueError("Text threshold must be between 0 and 1")
    
    if args.min_frame_gap < 1:
        raise ValueError("Minimum frame gap must be positive")
    
    if args.analyze_interval < 1:
        raise ValueError("Analysis interval must be positive")
    
    if args.start_time and args.start_time < 0:
        raise ValueError("Start time must be non-negative")
    
    if args.end_time and args.end_time < 0:
        raise ValueError("End time must be non-negative")
    
    if args.start_time and args.end_time and args.start_time >= args.end_time:
        raise ValueError("Start time must be less than end time")

def process_video(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Process video with the given arguments
    """
    # Set up logging
    logger = utils.setup_logging(args.log_file)
    
    try:
        # Initialize extractor
        extractor = VideoFrameExtractor(
            scene_threshold=args.scene_threshold,
            text_threshold=args.text_threshold,
            min_frame_gap=args.min_frame_gap,
            output_dir=args.output_dir,
            openai_api_key=args.openai_key
        )
        
        # Process video
        if args.input.startswith(('http://', 'https://', 'www.')):
            logger.info(f"Downloading YouTube video: {args.input}")
            video_path = extractor.download_youtube_video(args.input)
        else:
            logger.info(f"Using local video file: {args.input}")
            video_path = args.input
        
        # Extract frames
        logger.info("Processing video and analyzing frames...")
        extractor.process_video(
            video_path,
            start_time=args.start_time,
            end_time=args.end_time,
            analyze_interval=args.analyze_interval
        )
        
        # Generate outputs
        pdf_path = os.path.join(args.output_dir, 'extracted_frames.pdf')
        logger.info(f"Generating PDF: {pdf_path}")
        extractor.generate_pdf(pdf_path)
        
        return {
            'video_path': video_path,
            'output_dir': args.output_dir,
            'total_frames': len(extractor.extracted_frames)
        }
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise

def main():
    """
    Main entry point
    """
    try:
        # Parse and validate arguments
        args = parse_arguments()
        validate_args(args)
        
        # Process video
        results = process_video(args)
        
        # Print summary
        analysis_path = os.path.join(args.output_dir, 'frame_analysis.json')
        utils.print_summary(
            metadata=utils.load_metadata(os.path.join(args.output_dir, 'metadata.json')),
            analysis_path=analysis_path
        )
        
        print(f"\nAll extracted frames and analysis are saved in: {args.output_dir}")
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()