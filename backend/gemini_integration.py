import os
import logging
import json
import uuid
import requests
from typing import Dict, List, Optional, Tuple, Any, Union
from model import T5Summarizer
from datetime import datetime

class ChatSession:
    """Manages a chat session with context about a news article."""
    
    def __init__(self, session_id: str, article_text: str, article_summary: str):
        """Initialize a new chat session.
        
        Args:
            session_id: Unique identifier for this chat session
            article_text: The full text of the news article
            article_summary: The summary of the news article
        """
        self.session_id = session_id
        self.article_text = article_text
        self.article_summary = article_summary
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        self.messages = []
        self.system_prompt = f"""You are a helpful assistant that can discuss the following news article. 
        Here's the article summary: {article_summary}
        
        If asked for details not in the summary, you can refer to the full article text.
        Always be factual and only discuss information that is present in the article.
        If asked about topics not related to this article, politely redirect the conversation back to the article.
        """
    
    def add_message(self, role: str, content: str):
        """Add a message to the chat history.
        
        Args:
            role: The role of the message sender ('user' or 'assistant')
            content: The message content
        """
        self.messages.append({"role": role, "content": content})
        self.last_activity = datetime.now()
    
    def get_chat_history(self) -> List[Dict[str, str]]:
        """Get the chat history.
        
        Returns:
            List of message dictionaries with 'role' and 'content' keys
        """
        return self.messages
    
    def get_formatted_history(self) -> List[Dict[str, str]]:
        """Get the chat history formatted for Gemini API.
        
        Returns:
            List of message dictionaries formatted for the Gemini API
        """
        # Start with the system prompt
        formatted_history = [{"role": "system", "content": self.system_prompt}]
        
        # Add the conversation history
        formatted_history.extend(self.messages)
        
        return formatted_history


class GeminiAIIntegration:
    """Integration with Google's Gemini AI for enhanced summarization capabilities."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Gemini AI integration.
        
        Args:
            api_key: The Gemini API key. If None, will try to load from environment variable.
        """
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            logging.warning("Gemini API key not found. Gemini features will be disabled. Set the GEMINI_API_KEY environment variable to enable.")
            self.is_available = False
            # Removed: raise ValueError("Gemini API key not found or invalid. Please check your API key configuration.")
        else:
            # Log API key presence (without revealing the actual key)
            logging.info("Gemini API key found. Attempting to configure Gemini integration.")
            try:
                # Configure Gemini client (replace with actual configuration if needed)
                # For now, just setting base URL and headers as placeholders
                self.base_url = "https://generativelanguage.googleapis.com/v1/models"
                self.model_name = "gemini-1.0-pro" # Using the correct model name
                self.headers = {
                    'Content-Type': 'application/json'
                    # API key is passed in the URL, not in headers for Gemini API
                }
                # Add debug logging for API key (masked for security)
                if self.api_key:
                    masked_key = self.api_key[:4] + "*" * (len(self.api_key) - 8) + self.api_key[-4:] if len(self.api_key) > 8 else "****"
                    logging.debug(f"Using API key: {masked_key}")
                # Attempt to configure the actual Gemini client
                try:
                    import google.generativeai as genai
                    genai.configure(api_key=self.api_key)
                    
                    # Verify connection by making a simple test request
                    try:
                        # Test the API key with a simple model list request
                        models = genai.list_models()
                        logging.info(f"Successfully connected to Gemini API. Found {len(models)} models.")
                        self.is_available = True
                        logging.info("Gemini AI configured successfully.")
                    except Exception as api_err:
                        logging.error(f"API key validation failed: {api_err}")
                        self.is_available = False
                        
                except ImportError:
                    logging.warning("google-generativeai package not found. Cannot verify Gemini API key.")
                    self.is_available = False # Mark as unavailable if package is missing
                except Exception as genai_err:
                    logging.error(f"Failed to configure or verify Gemini AI: {genai_err}")
                    self.is_available = False
                    # Do not raise an error here, just log and mark as unavailable

            except Exception as e:
                logging.error(f"Error initializing Gemini settings: {str(e)}")
                self.is_available = False
                # Do not raise an error here either

        self.chat_sessions: Dict[str, Any] = {} # Store chat history {session_id: history}
        self.active_chat_sessions = {}
        logging.info("Gemini AI integration initialized successfully")
    
    def generate_summary(self, text: str, length: str = "medium", tone: str = "neutral") -> Optional[str]:
        """Generate a summary using Gemini AI.
        
        Args:
            text: The text to summarize
            length: Summary length ('short', 'medium', or 'long')
            tone: Summary tone ('neutral', 'analytical', or 'key_points')
            
        Returns:
            Generated summary or None if API is not available
        """
        if not self.is_available:
            logging.warning("Gemini AI is not available. Please check your API key.")
            return None
            
        try:
            # Prepare prompt based on length and tone
            length_instructions = {
                "short": "Create a very concise summary in 1-2 sentences.",
                "medium": "Create a summary in 3-5 sentences capturing the main points.",
                "long": "Create a comprehensive summary covering all important details."
            }
            
            tone_instructions = {
                "neutral": "Use a neutral, objective tone.",
                "analytical": "Use an analytical tone that examines the key arguments and evidence.",
                "key_points": "Structure the summary as a list of key points."
            }
            
            prompt = f"""Summarize the following text:
            
            {text}
            
            {length_instructions.get(length, length_instructions['medium'])}
            {tone_instructions.get(tone, tone_instructions['neutral'])}
            """
            
            # Ensure API key is properly included in the URL
            url = f"{self.base_url}/{self.model_name}:generateContent?key={self.api_key}"
            payload = {
                "contents": {
                    "role": "user",
                    "parts":[{"text": prompt}]
                },
                "generationConfig": {
                    "temperature": 0.7,
                    "topK": 40,
                    "topP": 0.95,
                    "maxOutputTokens": 1024
                }
            }
            # Add detailed logging for request debugging
            logging.debug(f"Sending request to Gemini API: {url}")
            response = requests.post(url, headers=self.headers, json=payload)
            
            # Log response status code
            logging.debug(f"Gemini API response status code: {response.status_code}")
            
            # Handle HTTP errors
            if response.status_code != 200:
                error_msg = f"Gemini API HTTP error: {response.status_code} - {response.text}"
                logging.error(error_msg)
                self.is_available = False
                return None
                
            response_json = response.json()
            logging.debug(f"Gemini API response received: {str(response_json)[:100]}...")
            
            # Better error handling
            if 'error' in response_json:
                error_info = response_json['error']
                error_message = f"Gemini API error: {error_info.get('message', 'Unknown error')}"
                logging.error(error_message)
                self.is_available = False  # Mark as unavailable if API returns an error
                return None
                
            summary = response_json.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '').strip()
            
            return summary
            
        except Exception as e:
            logging.error(f"Error generating summary with Gemini AI: {str(e)}")
            return None
    
    def enhance_t5_summary(self, original_text: str, t5_summary: str) -> str:
        """Enhance a T5-generated summary using Gemini AI.
        
        Args:
            original_text: The original text that was summarized
            t5_summary: The summary generated by the T5 model
            
        Returns:
            Enhanced summary or original T5 summary if Gemini is not available
        """
        if not self.is_available:
            return t5_summary
            
        try:
            prompt = f"""I have a summary of a text, but I'd like you to enhance it by:
            1. Improving clarity and readability
            2. Ensuring factual accuracy
            3. Making it more coherent and well-structured
            
            Original text:
            {original_text[:1000]}... [truncated]
            
            Current summary:
            {t5_summary}
            
            Please provide an enhanced version of this summary without adding new information that isn't in the original text.
            """
            
            # Ensure API key is properly included in the URL
            url = f"{self.base_url}/{self.model_name}:generateContent?key={self.api_key}"
            payload = {
                "contents": {
                    "role": "user",
                    "parts":[{"text": prompt}]
                },
                "generationConfig": {
                    "temperature": 0.7,
                    "topK": 40,
                    "topP": 0.95,
                    "maxOutputTokens": 1024
                }
            }
            # Add detailed logging for request debugging
            logging.debug(f"Sending request to Gemini API: {url}")
            response = requests.post(url, headers=self.headers, json=payload)
            
            # Log response status code
            logging.debug(f"Gemini API response status code: {response.status_code}")
            
            # Handle HTTP errors
            if response.status_code != 200:
                error_msg = f"Gemini API HTTP error: {response.status_code} - {response.text}"
                logging.error(error_msg)
                self.is_available = False
                return None
                
            response_json = response.json()
            logging.debug(f"Gemini API response received: {str(response_json)[:100]}...")
            
            # Better error handling
            if 'error' in response_json:
                error_info = response_json['error']
                error_message = f"Gemini API error: {error_info.get('message', 'Unknown error')}"
                logging.error(error_message)
                return t5_summary  # Return original summary if API returns an error
                
            enhanced_summary = response_json.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '').strip()
            
            return enhanced_summary
            
        except Exception as e:
            logging.error(f"Error enhancing summary with Gemini AI: {str(e)}")
            return t5_summary
    
    def compare_summaries(self, original_text: str, t5_summary: str, gemini_summary: str) -> Dict[str, Any]:
        """Compare T5 and Gemini summaries and provide analysis.
        
        Args:
            original_text: The original text that was summarized
            t5_summary: The summary generated by the T5 model
            gemini_summary: The summary generated by Gemini AI
            
        Returns:
            Dictionary with comparison metrics and analysis
        """
        if not self.is_available:
            return {"error": "Gemini AI is not available for comparison"}
            
        try:
            prompt = f"""Compare these two summaries of the same text and analyze their strengths and weaknesses.
            Provide your analysis as a JSON object with the following structure:
            {{"t5_strengths": [], "t5_weaknesses": [], "gemini_strengths": [], "gemini_weaknesses": [], "overall_recommendation": ""}}
            
            Original text (excerpt):
            {original_text[:500]}... [truncated]
            
            T5 Summary:
            {t5_summary}
            
            Gemini Summary:
            {gemini_summary}
            """
            
            # Ensure API key is properly included in the URL
            url = f"{self.base_url}/{self.model_name}:generateContent?key={self.api_key}"
            payload = {
                "contents": {
                    "role": "user",
                    "parts":[{"text": prompt}]
                },
                "generationConfig": {
                    "temperature": 0.7,
                    "topK": 40,
                    "topP": 0.95,
                    "maxOutputTokens": 1024
                }
            }
            # Add detailed logging for request debugging
            logging.debug(f"Sending request to Gemini API: {url}")
            response = requests.post(url, headers=self.headers, json=payload)
            
            # Log response status code
            logging.debug(f"Gemini API response status code: {response.status_code}")
            
            # Handle HTTP errors
            if response.status_code != 200:
                error_msg = f"Gemini API HTTP error: {response.status_code} - {response.text}"
                logging.error(error_msg)
                self.is_available = False
                return None
                
            response_json = response.json()
            logging.debug(f"Gemini API response received: {str(response_json)[:100]}...")
            
            # Better error handling
            if 'error' in response_json:
                error_info = response_json['error']
                error_message = f"Gemini API error: {error_info.get('message', 'Unknown error')}"
                logging.error(error_message)
                return {"error": error_message}
                
            analysis_text = response_json.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '').strip()
            
            # Extract JSON from response (handling potential formatting issues)
            try:
                # Try to parse the entire response as JSON
                analysis = json.loads(analysis_text)
            except json.JSONDecodeError:
                # If that fails, try to extract JSON using regex
                import re
                json_match = re.search(r'\{.*\}', analysis_text, re.DOTALL)
                if json_match:
                    try:
                        analysis = json.loads(json_match.group(0))
                    except json.JSONDecodeError:
                        # If still failing, return a simplified analysis
                        analysis = {
                            "error": "Could not parse analysis as JSON",
                            "raw_response": analysis_text
                        }
                else:
                    analysis = {
                        "error": "Could not extract JSON from response",
                        "raw_response": analysis_text
                    }
            
            return analysis
            
        except Exception as e:
            logging.error(f"Error comparing summaries with Gemini AI: {str(e)}")
            return {"error": str(e)}
            
    def create_chat_session(self, article_text: str, article_summary: str) -> str:
        """Create a new chat session for discussing a news article.
        
        Args:
            article_text: The full text of the news article
            article_summary: The summary of the news article
            
        Returns:
            Session ID for the new chat session
        """
        if not self.is_available:
            logging.warning("Gemini AI is not available. Cannot create chat session.")
            return None
            
        session_id = str(uuid.uuid4())
        self.active_chat_sessions[session_id] = ChatSession(session_id, article_text, article_summary)
        logging.info(f"Created new chat session with ID: {session_id}")
        return session_id
    
    def get_chat_session(self, session_id: str) -> Optional[ChatSession]:
        """Get a chat session by ID.
        
        Args:
            session_id: The ID of the chat session to retrieve
            
        Returns:
            ChatSession object or None if not found
        """
        return self.active_chat_sessions.get(session_id)
    
    def delete_chat_session(self, session_id: str) -> bool:
        """Delete a chat session.
        
        Args:
            session_id: The ID of the chat session to delete
            
        Returns:
            True if deleted, False if not found
        """
        if session_id in self.active_chat_sessions:
            del self.active_chat_sessions[session_id]
            logging.info(f"Deleted chat session with ID: {session_id}")
            return True
        return False
    
    def chat_with_article(self, session_id: str, user_message: str) -> Dict[str, Any]:
        """Send a message to the chatbot and get a response.
        
        Args:
            session_id: The ID of the chat session
            user_message: The user's message
            
        Returns:
            Dictionary with response text and metadata
        """
        if not self.is_available:
            return {"error": "Gemini AI is not available for chat"}
            
        chat_session = self.get_chat_session(session_id)
        if not chat_session:
            return {"error": f"Chat session with ID {session_id} not found"}
            
        try:
            # Add user message to chat history
            chat_session.add_message("user", user_message)
            
            # Get formatted chat history for Gemini
            chat_history = chat_session.get_formatted_history()
            
            # Generate response from Gemini
            # Ensure API key is properly included in the URL
            url = f"{self.base_url}/{self.model_name}:generateContent?key={self.api_key}"
            
            # Format the chat history properly for Gemini API
            formatted_messages = []
            for message in chat_history:
                formatted_messages.append({
                    "role": message["role"],
                    "parts": [{"text": message["content"]}]
                })
                
            payload = {
                "contents": formatted_messages,
                "generationConfig": {
                    "temperature": 0.7,
                    "topK": 40,
                    "topP": 0.95,
                    "maxOutputTokens": 1024
                }
            }
            # Add detailed logging for request debugging
            logging.debug(f"Sending request to Gemini API: {url}")
            response = requests.post(url, headers=self.headers, json=payload)
            
            # Log response status code
            logging.debug(f"Gemini API response status code: {response.status_code}")
            
            # Handle HTTP errors
            if response.status_code != 200:
                error_msg = f"Gemini API HTTP error: {response.status_code} - {response.text}"
                logging.error(error_msg)
                self.is_available = False
                return None
                
            response_json = response.json()
            logging.debug(f"Gemini API response received: {str(response_json)[:100]}...")
            
            # Better error handling
            if 'error' in response_json:
                error_info = response_json['error']
                error_message = f"Gemini API error: {error_info.get('message', 'Unknown error')}"
                logging.error(error_message)
                return {"error": error_message}
                
            assistant_message = response_json.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '').strip()
            
            # Add assistant response to chat history
            chat_session.add_message("assistant", assistant_message)
            
            return {
                "response": assistant_message,
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "chat_history": chat_session.get_chat_history()
            }
            
        except Exception as e:
            error_msg = f"Error generating chat response: {str(e)}"
            logging.error(error_msg)
            return {"error": error_msg}

class HybridSummarizer:
    """Hybrid summarization using both T5 and Gemini models."""
    
    def __init__(self, t5_model: T5Summarizer, gemini_integration: GeminiAIIntegration):
        """Initialize the hybrid summarizer.
        
        Args:
            t5_model: The T5Summarizer instance
            gemini_integration: The GeminiAIIntegration instance
        """
        self.t5_model = t5_model
        self.gemini = gemini_integration
        self.is_hybrid_available = self.gemini.is_available
        
    def generate_summary(self, text: str, mode: str = "hybrid", length: str = "medium", 
                         tone: str = "neutral", post_process: bool = True,
                         post_process_level: str = "basic") -> Dict[str, Any]:
        """Generate a summary using the specified mode.
        
        Args:
            text: The text to summarize
            mode: Summarization mode ('t5', 'gemini', or 'hybrid')
            length: Summary length ('short', 'medium', or 'long')
            tone: Summary tone ('neutral', 'analytical', or 'key_points')
            post_process: Whether to apply post-processing
            post_process_level: Post-processing level
            
        Returns:
            Dictionary with summaries and metadata
        """
        result = {"original_text_length": len(text)}
        
        # Generate T5 summary
        t5_summary = self.t5_model.generate_summary(
            text=text,
            length=length,
            tone=tone,
            post_process=post_process,
            post_process_level=post_process_level
        )
        result["t5_summary"] = t5_summary
        
        # If mode is t5 only, return early
        if mode == "t5":
            result["final_summary"] = t5_summary
            return result
            
        # Check if Gemini is available for other modes
        if not self.is_hybrid_available:
            logging.warning("Gemini AI is not available. Falling back to T5 summary.")
            result["final_summary"] = t5_summary
            result["fallback_to_t5"] = True
            return result
            
        # Generate Gemini summary
        gemini_summary = self.gemini.generate_summary(text, length, tone)
        result["gemini_summary"] = gemini_summary
        
        # If mode is gemini only, return early
        if mode == "gemini":
            result["final_summary"] = gemini_summary
            return result
            
        # For hybrid mode, enhance the T5 summary with Gemini
        if mode == "hybrid":
            enhanced_summary = self.gemini.enhance_t5_summary(text, t5_summary)
            result["final_summary"] = enhanced_summary
            
            # Add comparison analysis
            result["comparison"] = self.gemini.compare_summaries(text, t5_summary, gemini_summary)
            
        return result