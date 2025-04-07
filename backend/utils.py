import requests
from bs4 import BeautifulSoup
from typing import Dict, Optional, Tuple, Union
import re

def fetch_url_content(url: str) -> Tuple[bool, Union[str, Dict]]:
    """Fetch and extract the main content from a URL.
    
    Args:
        url: The URL to fetch content from
        
    Returns:
        A tuple containing (success_status, content_or_error)
    """
    try:
        # Send a GET request to the URL
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script_or_style in soup(["script", "style", "header", "footer", "nav"]):
            script_or_style.decompose()
        
        # Extract the main content
        # First, try to find the main article content
        article = soup.find('article') or soup.find(attrs={'class': re.compile(r'article|post|content', re.I)})
        
        if article:
            # If we found an article element, use its text
            content = article.get_text(separator=' ', strip=True)
        else:
            # Otherwise, use the body text
            content = soup.body.get_text(separator=' ', strip=True)
        
        # Clean up the content
        content = clean_text(content)
        
        if not content or len(content) < 100:  # If content is too short, it's probably not useful
            return False, {"error": "Could not extract meaningful content from the URL"}
        
        return True, content
        
    except requests.exceptions.RequestException as e:
        return False, {"error": f"Error fetching URL: {str(e)}"}
    except Exception as e:
        return False, {"error": f"Error processing content: {str(e)}"}

def clean_text(text: str) -> str:
    """Clean and normalize text for summarization.
    
    Args:
        text: The text to clean
        
    Returns:
        Cleaned text
    """
    # Replace multiple whitespace with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters and digits (optional, depending on your needs)
    # text = re.sub(r'[^\w\s]', '', text)
    # text = re.sub(r'\d+', '', text)
    
    return text.strip()

def validate_input(text: Optional[str] = None, url: Optional[str] = None) -> Tuple[bool, Union[str, Dict]]:
    """Validate the input text or URL.
    
    Args:
        text: The input text (optional)
        url: The input URL (optional)
        
    Returns:
        A tuple containing (is_valid, text_or_error)
    """
    # Check if at least one of text or URL is provided
    if not text and not url:
        return False, {"error": "Either text or URL must be provided"}
    
    # If URL is provided, fetch its content
    if url:
        is_valid, content = fetch_url_content(url)
        if not is_valid:
            return False, content  # Return the error
        return True, content
    
    # If text is provided, validate it
    if len(text) < 50:
        return False, {"error": "Text is too short for summarization (minimum 50 characters)"}
    
    # Clean the text
    cleaned_text = clean_text(text)
    
    return True, cleaned_text