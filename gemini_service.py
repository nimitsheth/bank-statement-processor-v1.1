# gemini_service.py
"""
Service module for Gemini API integration and AI processing
"""

import logging
import google.generativeai as genai
from config import GEMINI_API_KEY, GEMINI_MODEL_NAME, GEMINI_TEMPERATURE


class GeminiService:
    """Service class for handling Gemini API interactions"""
    
    def __init__(self):
        self.api_key = GEMINI_API_KEY
        self.model_name = GEMINI_MODEL_NAME
        self.temperature = GEMINI_TEMPERATURE
        self._configure_gemini()
    
    def _configure_gemini(self):
        """Configure Gemini API with credentials"""
        if not self.api_key:
            raise ValueError("Please set the GEMINI_API_KEY environment variable")
        
        genai.configure(api_key=self.api_key, transport="rest")
        logging.info("Gemini API configured successfully")
    
    def send_to_gemini(self, file_paths, prompt_text):
        """
        Send files and prompt to Gemini for processing
        
        Args:
            file_paths: Either a single file path (str) or list/tuple of file paths
            prompt_text: The prompt text to send with the files
            
        Returns:
            str: Response text from Gemini
        """
        logging.info("Preparing to send to Gemini: %s", file_paths)
        
        model = genai.GenerativeModel(
            self.model_name,
            generation_config={
                "temperature": self.temperature,
                "top_p": 1,
                "top_k": 40,
                # "max_output_tokens": 1024
            }
        )
        
        # Upload files - handle single file or multiple files
        uploaded_objs = self._upload_files(file_paths)
        
        # Create payload with uploaded files and prompt
        payload = uploaded_objs + [prompt_text]
        
        logging.info("Calling Gemini model.generate_content with %d payload items", len(payload))
        response = model.generate_content(payload)
        logging.info("Gemini response received")
        # logging.info("Gemini response: %s", response.text)
        return response.text
    
    def _upload_files(self, file_paths):
        """
        Upload files to Gemini
        
        Args:
            file_paths: Either a single file path (str) or list/tuple of file paths
            
        Returns:
            list: List of uploaded file objects
        """
        uploaded_objs = []
        
        if isinstance(file_paths, (list, tuple)):
            for path in file_paths:
                logging.info("Uploading file to Gemini: %s", path)
                uploaded = genai.upload_file(path)
                uploaded_objs.append(uploaded)
        else:
            logging.info("Uploading single file to Gemini: %s", file_paths)
            uploaded = genai.upload_file(file_paths)
            uploaded_objs.append(uploaded)
        
        return uploaded_objs