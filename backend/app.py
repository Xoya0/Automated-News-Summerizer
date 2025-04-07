import sys
import os

# Add the backend directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException, Form, Request, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, Union, List
import uvicorn
import json
import tempfile

from model import T5Summarizer
from utils import validate_input

# Initialize FastAPI app
app = FastAPI(title="T5 Summarizer API", description="API for text summarization using T5 model")

# Add CORS middleware to allow cross-origin requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
import asyncio

class TimeoutMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, timeout=60):
        super().__init__(app)
        self.timeout = timeout

    async def dispatch(self, request, call_next):
        try:
            return await asyncio.wait_for(call_next(request), timeout=self.timeout)
        except asyncio.TimeoutError:
            return Response(
                content=json.dumps({"detail": "Request processing timed out. Please try with shorter text or different parameters."}),
                status_code=504,
                media_type="application/json"
            )

app.add_middleware(TimeoutMiddleware, timeout=60)

try:
    print("Initializing T5 summarizer model...")
    summarizer = T5Summarizer(model_name="t5-small")
    print("T5 summarizer model initialized successfully")
except Exception as e:
    print(f"Failed to initialize T5 summarizer model: {str(e)}")
    raise RuntimeError(f"Model initialization failed: {str(e)}")

# Define request models
class SummarizationRequest(BaseModel):
    text: Optional[str] = None
    url: Optional[str] = None
    length: str = Field(default="medium", description="Summary length: 'short', 'medium', or 'long'")
    tone: str = Field(default="neutral", description="Summary tone: 'neutral', 'analytical', or 'key_points'")
    post_process: bool = Field(default=True, description="Whether to apply post-processing to the summary")
    post_process_level: str = Field(default="basic", description="Post-processing level: 'basic', 'medium', or 'advanced'")
    
    class Config:
        schema_extra = {
            "example": {
                "text": "This is a sample text to summarize...",
                "url": None,
                "length": "medium",
                "tone": "neutral",
                "post_process": True,
                "post_process_level": "basic"
            }
        }

class FeedbackRequest(BaseModel):
    summary_id: str
    rating: int = Field(..., ge=1, le=5, description="Rating from 1 to 5")
    comments: Optional[str] = None

# Define response models
class SummarizationResponse(BaseModel):
    summary: str
    summary_id: str
    original_text_length: int
    summary_length: int
    parameters: Dict[str, Any]

# API endpoints
@app.get("/")
async def root():
    return {"message": "Welcome to the T5 Summarizer API"}

@app.post("/summarize", response_model=SummarizationResponse)
async def summarize(request: SummarizationRequest):
    try:
        # Validate input (text or URL)
        is_valid, content_or_error = validate_input(request.text, request.url)
        
        if not is_valid:
            print(f"Input validation failed: {content_or_error['error']}")
            raise HTTPException(status_code=400, detail=content_or_error["error"])
        
        print(f"Processing text of length {len(content_or_error)} with parameters: length={request.length}, tone={request.tone}")
        
        # Generate summary with customized parameters
        try:
            summary = summarizer.generate_summary(
                text=content_or_error,
                length=request.length,
                tone=request.tone,
                post_process=request.post_process,
                post_process_level=request.post_process_level
            )
            
            # Create a unique ID for the summary (for feedback purposes)
            import hashlib
            import time
            summary_id = hashlib.md5(f"{content_or_error}{time.time()}".encode()).hexdigest()
            
            response_data = {
                "summary": summary,
                "summary_id": summary_id,
                "original_text_length": len(content_or_error),
                "summary_length": len(summary),
                "parameters": {
                    "length": request.length,
                    "tone": request.tone
                }
            }
            
            print(f"Successfully generated summary of length {len(summary)}")
            return response_data
            
        except Exception as e:
            error_msg = f"Error generating summary: {str(e)}"
            print(f"Summary generation failed: {error_msg}")
            raise HTTPException(status_code=500, detail=error_msg)
            
    except HTTPException as he:
        # Re-raise HTTP exceptions with their original status code
        raise he
    except Exception as e:
        # Handle any unexpected errors
        error_msg = f"Unexpected error during summarization: {str(e)}"
        print(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    # In a production system, you would store this feedback in a database
    # For now, we'll just acknowledge receipt
    print(f"Received feedback for summary {request.summary_id}: Rating {request.rating}")
    if request.comments:
        print(f"Comments: {request.comments}")
    
    return {"message": "Feedback received, thank you!"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Define fine-tuning request model
class FineTuningRequest(BaseModel):
    train_data: List[Dict[str, str]]
    eval_data: Optional[List[Dict[str, str]]] = None
    output_dir: str = Field(default="./fine_tuned_model", description="Directory to save the fine-tuned model")
    num_train_epochs: int = Field(default=3, description="Number of training epochs")
    clean_data: bool = Field(default=True, description="Whether to apply data cleaning to remove noisy examples")
    noise_threshold: float = Field(default=0.8, description="Threshold for identifying noisy examples (higher = more strict)")
    
    class Config:
        schema_extra = {
            "example": {
                "train_data": [
                    {"text": "This is a sample text to summarize...", "summary": "Sample summary."}
                ],
                "eval_data": [
                    {"text": "This is an evaluation text...", "summary": "Evaluation summary."}
                ],
                "output_dir": "./fine_tuned_model",
                "num_train_epochs": 3,
                "clean_data": True,
                "noise_threshold": 0.8
            }
        }

@app.post("/fine-tune")
async def fine_tune(request: FineTuningRequest):
    try:
        # Validate input data
        if not request.train_data or len(request.train_data) == 0:
            raise HTTPException(status_code=400, detail="Training data is required")
            
        for example in request.train_data:
            if "text" not in example or "summary" not in example:
                raise HTTPException(status_code=400, detail="Each training example must contain 'text' and 'summary' fields")
        
        # Create output directory if it doesn't exist
        os.makedirs(request.output_dir, exist_ok=True)
        
        # Start fine-tuning in a background task
        # In a production environment, you would use a task queue like Celery
        # For simplicity, we'll just start the process and return a response
        try:
            summarizer.fine_tune(
                train_data=request.train_data,
                eval_data=request.eval_data,
                output_dir=request.output_dir,
                num_train_epochs=request.num_train_epochs,
                clean_data=request.clean_data,
                noise_threshold=request.noise_threshold
            )
            
            return {"message": "Fine-tuning completed successfully", "output_dir": request.output_dir}
            
        except Exception as e:
            error_msg = f"Error during fine-tuning: {str(e)}"
            print(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)
            
    except HTTPException as he:
        # Re-raise HTTP exceptions with their original status code
        raise he
    except Exception as e:
        # Handle any unexpected errors
        error_msg = f"Unexpected error during fine-tuning: {str(e)}"
        print(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/upload-training-data")
async def upload_training_data(file: UploadFile = File(...)):
    try:
        # Create a temporary file to store the uploaded data
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as temp_file:
            # Write the uploaded file content to the temporary file
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Read the JSON data from the temporary file
        try:
            with open(temp_file_path, "r") as f:
                data = json.load(f)
            
            # Validate the data format
            if not isinstance(data, list):
                raise HTTPException(status_code=400, detail="Training data must be a JSON array")
                
            for example in data:
                if not isinstance(example, dict) or "text" not in example or "summary" not in example:
                    raise HTTPException(status_code=400, detail="Each training example must contain 'text' and 'summary' fields")
            
            # Clean up the temporary file
            os.unlink(temp_file_path)
            
            return {"message": f"Successfully uploaded {len(data)} training examples", "data_preview": data[:3]}
            
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON format")
            
    except Exception as e:
        error_msg = f"Error processing uploaded file: {str(e)}"
        print(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

# Run the API server when the script is executed directly
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)