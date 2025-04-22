import logging
import re
import nltk
import string
from typing import List, Dict, Optional, Tuple, Any
from datasets import load_dataset
from torch.utils.data import Dataset

# Ensure NLTK resources are available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class CNNDailyMailDataLoader:
    def __init__(self):
        """Initialize the CNN/DailyMail dataset loader."""
        logging.info("Initializing CNN/DailyMail dataset loader")
        self.dataset = None

    def load_dataset(self, split: str = 'train') -> Optional[List[Dict[str, str]]]:
        """Load the CNN/DailyMail dataset from Hugging Face.
        
        Args:
            split: Dataset split to load ('train', 'validation', or 'test')
            
        Returns:
            List of dictionaries with 'text' and 'summary' keys
        """
        try:
            logging.info(f"Loading CNN/DailyMail dataset ({split} split)")
            self.dataset = load_dataset('cnn_dailymail', '3.0.0', split=split)
            
            if not self.dataset:
                logging.error("Failed to load dataset")
                return None
                
            # Convert to expected format with preprocessing
            formatted_data = []
            for item in self.dataset:
                # Clean and preprocess the article and summary
                cleaned_article = self._clean_text(item['article'])
                cleaned_summary = self._clean_text(item['highlights'])
                
                # Skip examples with empty text or summary after cleaning
                if not cleaned_article or not cleaned_summary:
                    continue
                    
                formatted_data.append({
                    'text': cleaned_article,
                    'summary': cleaned_summary
                })
                
            logging.info(f"Successfully loaded and preprocessed {len(formatted_data)} examples")
            return formatted_data
            
        except Exception as e:
            logging.error(f"Error loading dataset: {str(e)}")
            return None
            
    def _clean_text(self, text: str) -> str:
        """Clean and preprocess text data.
        
        Args:
            text: The text to clean
            
        Returns:
            Cleaned text
        """
        if not text or not isinstance(text, str):
            return ""
            
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common news article artifacts
        patterns = [
            r'\([^)]*\) - ',  # (Reuters) -, (AP) -
            r'\[[^\]]*\]',     # [VIDEO], [AUDIO], etc.
            r'Share (?:Save|this article|on)',
            r'(?:Follow us|Subscribe) (?:on|to)',
            r'Read (?:more|Full Article):?',
            r'(?:Photo|Image|Credit|Source):',
            r'Advertisement|Sponsored Content',
            r'Click here to view related media',
            r'https?://\S+',    # URLs
            r'@\w+',           # Twitter handles
            r'#\w+',           # Hashtags
            r'\d+\s*(?:st|nd|rd|th)\s*(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s*\d{4}', # Dates
            r'\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)' # Times
        ]
        
        for pattern in patterns:
            text = re.sub(pattern, '', text)
        
        # Fix common punctuation issues
        text = re.sub(r'\s+([.,;:!?])', r'\1', text)  # Remove space before punctuation
        text = re.sub(r'([.,;:!?])([A-Za-z])', r'\1 \2', text)  # Add space after punctuation
        
        # Remove duplicate punctuation
        text = re.sub(r'([.,;:!?])[.,;:!?]+', r'\1', text)
        
        # Normalize sentence endings
        text = re.sub(r'\s*\.\s*', '. ', text)
        
        # Ensure proper capitalization of sentences
        sentences = nltk.sent_tokenize(text)
        sentences = [s[0].upper() + s[1:] if s and len(s) > 0 else s for s in sentences]
        text = ' '.join(sentences)
        
        # Remove any remaining non-printable characters
        text = ''.join(c for c in text if c in string.printable)
        
        return text.strip()

    def get_batch(self, data: List[Dict[str, str]], batch_size: int, start_idx: int) -> List[Dict[str, str]]:
        """Get a batch of examples from the dataset.
        
        Args:
            data: List of all examples
            batch_size: Size of batch to return
            start_idx: Starting index for the batch
            
        Returns:
            List of examples for the batch
        """
        end_idx = min(start_idx + batch_size, len(data))
        return data[start_idx:end_idx]