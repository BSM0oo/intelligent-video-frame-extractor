# Video Frame Extractor

An intelligent video frame extraction tool that uses computer vision and LLM-powered analysis to capture key frames from videos, particularly useful for educational content, presentations, and lectures.

## Features

- 🎥 Automatic key frame extraction from videos (local files or YouTube)
- 🧠 LLM-powered frame analysis using GPT-4 Vision
- 🔍 Real-time content detection and analysis
- 📊 Scene change detection using multiple metrics
- 📝 OCR text extraction and comparison
- 🖼️ Automatic missing content detection
- 📄 PDF generation of extracted frames
- 📊 Detailed JSON analysis reports

## Directory Structure

```
video_frame_extractor/
├── README.md           # Comprehensive documentation
├── __init__.py        # Package initialization
├── cli.py             # Command-line interface
├── extractor.py       # Main extractor class
├── frame_analyzer.py  # Frame analysis and CV operations
├── llm_analyzer.py    # LLM-based content analysis
├── requirements.txt   # Package dependencies
└── utils.py          # Utility functions

When running the tool, additional directories are created:
video_frame_extractor/
├── [all files above]
└── extracted_frames/  # (created during runtime)
    ├── frames/       # Extracted video frames
    ├── analysis/     # LLM analysis results
    ├── exports/      # Generated PDFs and other exports
    ├── metadata.json # Frame extraction metadata
    └── frame_analysis.json # Detailed LLM analysis
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/video-frame-extractor.git
cd video-frame-extractor
```

2. Install required Python packages:
```bash
pip install -r requirements.txt
```

3. System Requirements:
- Python 3.8+
- tesseract-ocr (for OCR capabilities)
- OpenAI API key (for LLM analysis)

## Project Structure

The project is organized into several modules:

### `extractor.py`
- Main class `VideoFrameExtractor`
- Handles high-level video processing and frame extraction
- Coordinates between different components
- Manages the extraction pipeline

### `frame_analyzer.py`
- Contains the `FrameAnalyzer` class
- Implements scene change detection algorithms
- Handles frame comparison and quality assessment
- OCR text extraction and comparison

### `llm_analyzer.py`
- Manages LLM-based frame analysis
- Implements GPT-4 Vision integration
- Handles missing content detection
- Generates analysis reports

### `utils.py`
- Utility functions for image processing
- File handling helpers
- Logging setup
- Data structure conversions

### `cli.py`
- Command-line interface implementation
- Argument parsing
- User interaction handling
- Progress reporting

## Usage

### Basic Usage

```bash
python cli.py -i "https://youtube.com/watch?v=example" \
    --output-dir "lecture_frames" \
    --openai-key "your_api_key" \
    --analyze-interval 50
```

### Configuration Options

- `--input, -i`: YouTube URL or local video path
- `--output-dir, -o`: Output directory (default: 'extracted_frames')
- `--openai-key`: OpenAI API key (can also be set via OPENAI_API_KEY env variable)
- `--analyze-interval`: Number of frames between LLM analyses (default: 50)
- `--scene-threshold`: Threshold for scene change detection (default: 0.7)
- `--text-threshold`: Threshold for text change detection (default: 0.8)
- `--min-frame-gap`: Minimum frames between extractions (default: 15)

### Advanced Usage Examples

1. Process specific section of a video:
```bash
python cli.py -i "lecture.mp4" \
    --start-time 300 \
    --end-time 600 \
    --analyze-interval 25
```

2. Higher sensitivity for presentation slides:
```bash
python cli.py -i "presentation.mp4" \
    --scene-threshold 0.6 \
    --text-threshold 0.9 \
    --min-frame-gap 30
```

## Function Documentation

### VideoFrameExtractor Class

#### `__init__(self, **kwargs)`
Initializes the frame extractor with configurable parameters:
- `scene_threshold`: Sensitivity to visual changes (0-1)
- `text_threshold`: Sensitivity to text changes (0-1)
- `min_frame_gap`: Minimum frames between extractions
- `output_dir`: Directory for saving outputs
- `openai_api_key`: API key for LLM analysis

#### `process_video(self, video_path, **kwargs)`
Main video processing function:
- Extracts frames based on scene/text changes
- Performs periodic LLM analysis
- Automatically identifies missing content
- Generates analysis reports

#### `analyze_frames_with_llm(self, frames_subset=None)`
LLM-based frame analysis:
- Analyzes visual elements and content
- Identifies key concepts and topics
- Detects missing context or transitions
- Assesses frame quality and completeness

#### `review_and_extract_missing(self, video_path, **kwargs)`
Handles missing content extraction:
- Reviews existing frames
- Identifies gaps in content
- Extracts additional frames as needed
- Updates analysis reports

### FrameAnalyzer Class

#### `calculate_histogram_difference(self, frame1, frame2)`
Computes color histogram differences between frames

#### `detect_edges(self, frame)`
Performs edge detection for visual change analysis

#### `assess_frame_quality(self, frame)`
Evaluates frame quality (blur, contrast, etc.)

## Development and Contribution

### Future Improvements

1. Technical Enhancements:
- Implement parallel processing for faster extraction
- Add support for more video formats
- Improve memory efficiency for large videos
- Add batch processing capabilities
- Implement video segment classification
- Add support for audio analysis

2. Analysis Features:
- Add support for multiple LLM providers
- Implement custom OCR models
- Add support for diagram/chart recognition
- Implement content summarization
- Add speaker detection/recognition
- Implement gesture recognition

3. User Experience:
- Create web interface
- Add real-time preview capabilities
- Implement frame editing features
- Add custom PDF template support
- Create interactive analysis reports
- Add batch processing interface

4. Integration Possibilities:
- Add integration with cloud storage services
- Implement API endpoints
- Add support for video streaming platforms
- Create plugins for video editing software
- Add support for live stream processing

### Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for GPT-4 Vision API
- OpenCV community
- tesseract-ocr project
- youtube-dl developers

## Support

For support, please open an issue in the GitHub repository or contact the maintainers.