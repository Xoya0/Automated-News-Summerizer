import React, { useState } from 'react';
import { Container, Typography, Box, TextField, Button, Paper, CircularProgress, Divider, FormControl, InputLabel, Select, MenuItem, Tooltip, IconButton, Rating, Snackbar, Alert, ThemeProvider, createTheme, CssBaseline, Tab, Tabs } from '@mui/material';
import { HelpOutline, ContentCopy, FormatListBulleted, Article, DarkMode, LightMode, Chat } from '@mui/icons-material';
import axios from 'axios';
import ChatComponent from './components/ChatComponent';

// Create theme instances
const lightTheme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#2196f3',
    },
    secondary: {
      main: '#1976d2',
    },
    background: {
      default: '#f5f7fa',
      paper: '#ffffff',
    },
  },
  typography: {
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
  },
  shape: {
    borderRadius: 12,
  },
});

const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#90caf9',
    },
    secondary: {
      main: '#64b5f6',
    },
    background: {
      default: '#1a1a1a',
      paper: '#2d3436',
    },
  },
  typography: {
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
  },
  shape: {
    borderRadius: 12,
  },
});

// Configure axios defaults
axios.defaults.headers.common['Content-Type'] = 'application/json';
axios.defaults.timeout = 60000; // 60 seconds timeout - increased from 15 seconds to handle longer processing times

// Add axios interceptors for better error handling
axios.interceptors.response.use(
  response => response,
  error => {
    console.error('API Error:', error);
    if (error.code === 'ECONNABORTED') {
      throw new Error('Request timeout. Please try again later.');
    }
    if (error.code === 'ERR_NETWORK') {
      throw new Error('Unable to connect to the server. Please check if the backend service is running on port 8000.');
    }
    if (!error.response) {
      throw new Error('Network error. Please check your internet connection.');
    }
    if (error.response.status === 404) {
      throw new Error('API endpoint not found. Please check the server configuration.');
    }
    if (error.response.status === 500) {
      throw new Error('Server error. Please try again later.');
    }
    throw error.response?.data?.detail || error.message || 'An unexpected error occurred';
  }
);

function App() {
  // State variables
  const [inputType, setInputType] = useState('text');
  const [inputText, setInputText] = useState('');
  const [inputUrl, setInputUrl] = useState('');
  const [summaryLength, setSummaryLength] = useState('medium');
  const [summaryTone, setSummaryTone] = useState('neutral');
  const [outputFormat, setOutputFormat] = useState('paragraph');
  const [summary, setSummary] = useState('');
  const [summaryId, setSummaryId] = useState('');
  const [chatSessionId, setChatSessionId] = useState('');
  const [loading, setLoading] = useState(false);
  const [showFeedback, setShowFeedback] = useState(false);
  const [feedbackRating, setFeedbackRating] = useState(0);
  const [feedbackComments, setFeedbackComments] = useState('');
  const [snackbarOpen, setSnackbarOpen] = useState(false);
  const [snackbarMessage, setSnackbarMessage] = useState('');
  const [snackbarSeverity, setSnackbarSeverity] = useState('success');
  const [darkMode, setDarkMode] = useState(false);
  const [error, setError] = useState('');
  const [activeTab, setActiveTab] = useState(0);
  
  // Theme
  const theme = darkMode ? darkTheme : lightTheme;

  // Function to handle summarization request
  const handleSummarize = async () => {
    // Reset previous results
    setSummary('');
    setSummaryId('');
    setError('');
    setShowFeedback(false);
    
    // Validate input
    if (inputType === 'text' && inputText.trim().length < 50) {
      setError('Please enter at least 50 characters of text to summarize.');
      return;
    }
    
    if (inputType === 'url' && !inputUrl.trim()) {
      setError('Please enter a valid URL.');
      return;
    }
    
    // Prepare request data
    const requestData = {
      text: inputType === 'text' ? inputText : null,
      url: inputType === 'url' ? inputUrl : null,
      length: summaryLength,
      tone: summaryTone
    };
    
    setLoading(true);
    
    try {
      // Send request to API
      const response = await axios.post('/summarize', requestData);
      
      // Update state with response data
      setSummary(response.data.summary);
      setSummaryId(response.data.summary_id);
      
      // Ensure chatSessionId is properly set
      if (response.data.chat_session_id) {
        setChatSessionId(response.data.chat_session_id);
        console.log('Chat session ID received:', response.data.chat_session_id);
      } else {
        // If backend doesn't provide chat_session_id, create one using summary_id
        const generatedSessionId = `chat_${response.data.summary_id}`;
        setChatSessionId(generatedSessionId);
        console.log('Generated chat session ID:', generatedSessionId);
      }
      
      setShowFeedback(true);
      setActiveTab(0); // Switch to summary tab
      
      // Show success message
      setSnackbarMessage('Summary generated successfully!');
      setSnackbarSeverity('success');
      setSnackbarOpen(true);
    } catch (err) {
      // Handle error
      console.error('Error generating summary:', err);
      
      // Check for specific proxy connection errors
      if (err.code === 'ECONNREFUSED' || err.message.includes('ECONNREFUSED')) {
        setError('Could not connect to the backend server. Please ensure the backend service is running on port 8000.');
      } else if (err.response?.status === 404) {
        setError('Backend endpoint not found. Please check if the backend service is properly configured.');
      } else {
        setError(err.response?.data?.detail || 'An error occurred while generating the summary. Please try again.');
      }
      
      // Show error message
      setSnackbarMessage('Failed to generate summary');
      setSnackbarSeverity('error');
      setSnackbarOpen(true);
    } finally {
      setLoading(false);
    }
  };
  
  // Function to handle feedback submission
  const handleSubmitFeedback = async () => {
    if (!summaryId || feedbackRating === 0) return;
    
    try {
      await axios.post('/feedback', {
        summary_id: summaryId,
        rating: feedbackRating,
        comments: feedbackComments
      });
      
      // Reset feedback form
      setFeedbackRating(0);
      setFeedbackComments('');
      setShowFeedback(false);
      
      // Show success message
      setSnackbarMessage('Thank you for your feedback!');
      setSnackbarSeverity('success');
      setSnackbarOpen(true);
    } catch (err) {
      console.error('Error submitting feedback:', err);
      
      // Show error message
      setSnackbarMessage('Failed to submit feedback');
      setSnackbarSeverity('error');
      setSnackbarOpen(true);
    }
  };
  
  // Function to copy summary to clipboard
  const handleCopySummary = () => {
    navigator.clipboard.writeText(summary);
    setSnackbarMessage('Summary copied to clipboard');
    setSnackbarSeverity('info');
    setSnackbarOpen(true);
  };
  
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box className={darkMode ? 'dark-mode' : ''}>
        <Container maxWidth="md">
          {/* Hero Section */}
          <Box className="hero-section">
            <Typography variant="h3" component="h1" gutterBottom>
              T5 Summarizer
            </Typography>
            <Typography variant="h6" paragraph>
              Transform your text into concise, meaningful summaries with AI-powered precision
            </Typography>
            <IconButton
              onClick={() => setDarkMode(!darkMode)}
              sx={{ color: 'white', position: 'absolute', top: 16, right: 16 }}
            >
              {darkMode ? <LightMode /> : <DarkMode />}
            </IconButton>
          </Box>

          {/* Main Content */}
          <Box sx={{ my: 4 }}>
            {/* Input Section */}
            <Paper elevation={3} sx={{ p: 3, mb: 4 }}>
              <Typography variant="h5" gutterBottom sx={{ fontWeight: 600 }}>
                Input
              </Typography>
              
              {/* Input Type Selection */}
              <Box sx={{ mb: 3 }}>
                <Button 
                  variant={inputType === 'text' ? 'contained' : 'outlined'} 
                  onClick={() => setInputType('text')}
                  sx={{ mr: 2 }}
                >
                  <Article sx={{ mr: 1 }} /> Text
                </Button>
                <Button 
                  variant={inputType === 'url' ? 'contained' : 'outlined'} 
                  onClick={() => setInputType('url')}
                >
                  <FormatListBulleted sx={{ mr: 1 }} /> URL
                </Button>
              </Box>
              
              {/* Text Input */}
              {inputType === 'text' && (
                <TextField
                  label="Enter text to summarize"
                  multiline
                  rows={6}
                  fullWidth
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  variant="outlined"
                  placeholder="Paste or type your text here (minimum 50 characters)"
                  sx={{ mb: 3 }}
                />
              )}
              
              {/* URL Input */}
              {inputType === 'url' && (
                <TextField
                  label="Enter URL to summarize"
                  fullWidth
                  value={inputUrl}
                  onChange={(e) => setInputUrl(e.target.value)}
                  variant="outlined"
                  placeholder="https://example.com/article"
                  sx={{ mb: 3 }}
                />
              )}
              
              {/* Customization Options */}
              <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
                Customize Summary
              </Typography>
              
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3, mb: 3 }}>
                {/* Summary Length Selection */}
                <Box>
                  <Typography gutterBottom>Summary Length</Typography>
                  <FormControl fullWidth>
                    <Select
                      value={summaryLength}
                      onChange={(e) => setSummaryLength(e.target.value)}
                      displayEmpty
                    >
                      <MenuItem value="short">Short</MenuItem>
                      <MenuItem value="medium">Medium</MenuItem>
                      <MenuItem value="long">Long</MenuItem>
                    </Select>
                    <Box sx={{ display: 'flex', alignItems: 'center', mt: 0.5 }}>
                      <Typography variant="caption" color="text.secondary">
                        {summaryLength === 'short' ? 'Concise overview' :
                         summaryLength === 'medium' ? 'Balanced summary' :
                         'Detailed explanation'}
                      </Typography>
                      <Tooltip title="Choose how detailed you want your summary to be">
                        <IconButton size="small">
                          <HelpOutline fontSize="small" />
                        </IconButton>
                      </Tooltip>
                    </Box>
                  </FormControl>
                </Box>
                
                {/* Summary Tone and Format */}
                <Box sx={{ display: 'flex', flexDirection: { xs: 'column', sm: 'row' }, gap: 2 }}>
                  <FormControl fullWidth>
                    <InputLabel id="summary-tone-label">Tone</InputLabel>
                    <Select
                      labelId="summary-tone-label"
                      value={summaryTone}
                      label="Tone"
                      onChange={(e) => setSummaryTone(e.target.value)}
                    >
                      <MenuItem value="neutral">Neutral</MenuItem>
                      <MenuItem value="analytical">Analytical</MenuItem>
                      <MenuItem value="key_points">Key Points</MenuItem>
                    </Select>
                  </FormControl>

                  <FormControl fullWidth>
                    <InputLabel id="output-format-label">Format</InputLabel>
                    <Select
                      labelId="output-format-label"
                      value={outputFormat}
                      label="Format"
                      onChange={(e) => setOutputFormat(e.target.value)}
                    >
                      <MenuItem value="paragraph">Paragraph</MenuItem>
                      <MenuItem value="bullet_points">Bullet Points</MenuItem>
                    </Select>
                  </FormControl>
                </Box>
              </Box>

              {/* Generate Button */}
              <Button
                variant="contained"
                onClick={handleSummarize}
                disabled={loading}
                fullWidth
                sx={{ mt: 2 }}
              >
                {loading ? (
                  <CircularProgress size={24} color="inherit" />
                ) : (
                  'Generate Summary'
                )}
              </Button>
            </Paper>

            {/* Summary and Chat Tabs */}
            {summary && (
              <>
                <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 2 }}>
                  <Tabs value={activeTab} onChange={(e, newValue) => setActiveTab(newValue)} aria-label="summary and chat tabs">
                    <Tab icon={<Article />} label="Summary" />
                    <Tab icon={<Chat />} label="Chat" disabled={!chatSessionId} />
                  </Tabs>
                </Box>
                
                {/* Summary Tab */}
                {activeTab === 0 && (
                  <Paper elevation={3} sx={{ p: 3, mb: 4 }} className="summary-container">
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                      <Typography variant="h5" sx={{ fontWeight: 600 }}>
                        Summary
                      </Typography>
                      <Tooltip title="Copy to clipboard">
                        <IconButton onClick={handleCopySummary}>
                          <ContentCopy />
                        </IconButton>
                      </Tooltip>
                    </Box>
                    <Divider sx={{ my: 2 }} />
                    <Typography variant="body1" sx={{ whiteSpace: 'pre-wrap' }}>
                      {summary}
                    </Typography>
                  </Paper>
                )}
                
                {/* Chat Tab */}
                {activeTab === 1 && (
                  <ChatComponent summaryId={summaryId} chatSessionId={chatSessionId} />
                )}
              </>
            )}

            {/* Feedback Section */}
            {showFeedback && (
              <Paper elevation={3} sx={{ p: 3 }}>
                <Typography variant="h5" gutterBottom sx={{ fontWeight: 600 }}>
                  Rate Your Summary
                </Typography>
                <Rating
                  value={feedbackRating}
                  onChange={(event, newValue) => setFeedbackRating(newValue)}
                  size="large"
                  sx={{ mb: 2 }}
                />
                <TextField
                  label="Comments (optional)"
                  multiline
                  rows={3}
                  fullWidth
                  value={feedbackComments}
                  onChange={(e) => setFeedbackComments(e.target.value)}
                  variant="outlined"
                  sx={{ mb: 2 }}
                />
                <Button
                  variant="contained"
                  onClick={handleSubmitFeedback}
                  disabled={feedbackRating === 0}
                  fullWidth
                >
                  Submit Feedback
                </Button>
              </Paper>
            )}
          </Box>
        </Container>
      </Box>

      {/* Snackbar for notifications */}
      <Snackbar
        open={snackbarOpen}
        autoHideDuration={6000}
        onClose={() => setSnackbarOpen(false)}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert
          onClose={() => setSnackbarOpen(false)}
          severity={snackbarSeverity}
          sx={{ width: '100%' }}
        >
          {snackbarMessage}
        </Alert>
      </Snackbar>
    </ThemeProvider>
  );
}

export default App;