# T5 Summarizer Application

A powerful text summarization application built with T5 transformer model, providing customizable summaries with different lengths, tones, and styles.

## Features

- Text summarization using state-of-the-art T5 model
- Customizable summary length (short, medium, long)
- Multiple tone options (neutral, analytical, key points)
- Advanced text preprocessing and post-processing
- GPU acceleration support
- Sentence variation generation
- Comprehensive evaluation metrics

## Tech Stack

### Backend
- Python 3.8+
- PyTorch
- Transformers (Hugging Face)
- NLTK
- FastAPI (for API endpoints)

### Frontend
- React.js
- Modern JavaScript (ES6+)
- CSS3

## Installation

### Backend Setup

1. Clone the repository:
```bash
git clone [your-repo-url]
cd T5-Summarizer
```

2. Install Python dependencies:
```bash
cd backend
pip install -r requirements.txt
```

3. Download required NLTK data (will be done automatically on first run)

### Frontend Setup

1. Install Node.js dependencies:
```bash
cd frontend
npm install
```

## Usage

### Starting the Backend Server

```bash
cd backend
python app.py
```

The backend server will start on http://localhost:8000

### Starting the Frontend Development Server

```bash
cd frontend
npm start
```

The frontend development server will start on http://localhost:3000

## API Endpoints

- POST `/summarize`
  - Input: JSON with text and parameters
  - Output: Generated summary

## Configuration

The application can be configured through the following parameters:

- `model_name`: T5 model variant (default: "t5-small")
- `learning_rate`: Optimization learning rate
- `batch_size`: Processing batch size
- `gradient_accumulation_steps`: Steps for gradient accumulation

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Hugging Face Transformers library
- T5 model developers
- NLTK contributors