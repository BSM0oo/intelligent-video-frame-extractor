from openai import OpenAI
from base64 import b64encode
import logging
from typing import List, Dict, Any, Optional
import json
import os

class LLMAnalyzer:
    """
    Handles LLM-based frame analysis and content detection
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the LLM analyzer
        
        Args:
            api_key: OpenAI API key
        """
        self.llm_client = OpenAI(api_key=api_key) if api_key else None
        self.logger = logging.getLogger(__name__)
        
        if not self.llm_client:
            self.logger.warning("No OpenAI API key provided. LLM analysis will be disabled.")
    
    def analyze_frame(self, frame_path: str, frame_info: Dict) -> Dict[str, Any]:
        """
        Analyze a single frame using GPT-4 Vision
        """
        if not self.llm_client:
            return {}
            
        try:
            # Read image and convert to base64
            with open(frame_path, 'rb') as img_file:
                image_data = b64encode(img_file.read()).decode()
            
            # Analyze image with GPT-4 Vision
            response = self.llm_client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Analyze this frame from an educational video. Identify:\n"
                                       "1. Main visual elements (diagrams, charts, text)\n"
                                       "2. Key concepts or topics being presented\n"
                                       "3. Any potential missing context or transitions\n"
                                       "4. Completeness of information (are parts of diagrams/text cut off?)\n"
                                       "5. Quality issues (blur, poor contrast, etc)"
                            },
                            {
                                "type": "image",
                                "image_url": f"data:image/jpeg;base64,{image_data}"
                            }
                        ]
                    }
                ],
                max_tokens=500
            )
            
            return {
                'timestamp': frame_info.get('timestamp'),
                'analysis': response.choices[0].message.content,
                'frame_path': frame_path
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing frame {frame_path}: {str(e)}")
            return {}
    
    def analyze_sequence(self, frame_analyses: List[Dict]) -> str:
        """
        Analyze a sequence of frames for continuity and missing content
        """
        if not self.llm_client or not frame_analyses:
            return ""
            
        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "user",
                        "content": f"Review these {len(frame_analyses)} sequential frames from an educational video:\n\n" +
                                  "\n\n".join([f"Frame at {a['timestamp']}s:\n{a['analysis']}" for a in frame_analyses]) +
                                  "\n\nIdentify:\n1. Any gaps in content or logical transitions\n" +
                                  "2. Potentially missing important frames\n3. Suggestions for additional frames to capture"
                    }
                ]
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            self.logger.error(f"Error analyzing sequence: {str(e)}")
            return ""
    
    def find_missing_timepoints(self, sequence_analysis: str) -> List[float]:
        """
        Parse sequence analysis to identify timestamps where additional frames might be needed
        """
        if not self.llm_client or not sequence_analysis:
            return []
            
        try:
            # Use GPT to extract specific timestamps from the analysis
            response = self.llm_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "user",
                        "content": f"Based on this analysis of video frames:\n\n{sequence_analysis}\n\n" +
                                  "Extract specific timestamps (in seconds) where additional frames should be captured. " +
                                  "Return only a comma-separated list of numbers."
                    }
                ]
            )
            
            # Parse the response into a list of timestamps
            timestamps_str = response.choices[0].message.content.strip()
            return [float(t) for t in timestamps_str.split(',') if t.strip()]
            
        except Exception as e:
            self.logger.error(f"Error parsing timestamps: {str(e)}")
            return []
    
    def save_analysis(self, output_dir: str, initial_analysis: Dict, final_analysis: Dict = None):
        """
        Save analysis results to JSON file
        """
        try:
            analysis_path = os.path.join(output_dir, 'frame_analysis.json')
            with open(analysis_path, 'w') as f:
                json.dump({
                    'initial_analysis': initial_analysis,
                    'final_analysis': final_analysis or {}
                }, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving analysis: {str(e)}")
