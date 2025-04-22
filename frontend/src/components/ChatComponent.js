import React, { useState, useEffect, useRef } from 'react';
import { Box, Paper, Typography, TextField, Button, CircularProgress, Divider, Avatar, IconButton } from '@mui/material';
import { Send, Person, SmartToy } from '@mui/icons-material';
import axios from 'axios';

const ChatComponent = ({ summaryId, chatSessionId }) => {
  const [message, setMessage] = useState('');
  const [chatHistory, setChatHistory] = useState([]);
  const [sessionId, setSessionId] = useState(chatSessionId || '');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const messagesEndRef = useRef(null);

  // Scroll to bottom of chat when messages change
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [chatHistory]);

  // Initialize chat session if summaryId is provided but no sessionId
  useEffect(() => {
    const initializeChat = async () => {
      if (summaryId && !sessionId) {
        try {
          setLoading(true);
          setError(''); // Clear any previous errors
          
          const formData = new FormData();
          formData.append('summary_id', summaryId);
          
          console.log('Initializing chat session with summary ID:', summaryId);
          
          const response = await axios.post('/chat/new', formData, {
            headers: {
              'Content-Type': 'multipart/form-data',
            },
          });
          
          if (response.data && response.data.session_id) {
            console.log('Chat session initialized with ID:', response.data.session_id);
            setSessionId(response.data.session_id);
          } else {
            console.error('No session ID returned from server');
            setError('Server did not return a valid session ID. Please try again.');
          }
        } catch (err) {
          console.error('Error initializing chat session:', err);
          let errorMessage = 'Failed to initialize chat session. Please try again.';
          
          if (err.response?.status === 404) {
            errorMessage = 'Chat initialization endpoint not found. Please check if the backend service is running.';
          } else if (err.response?.data?.detail) {
            errorMessage = err.response.data.detail;
          }
          
          setError(errorMessage);
        } finally {
          setLoading(false);
        }
      } else if (sessionId) {
        console.log('Using existing chat session ID:', sessionId);
      }
    };

    initializeChat();
  }, [summaryId, sessionId]);

  const handleSendMessage = async () => {
    if (!message.trim() || !sessionId) return;

    try {
      setLoading(true);
      setError('');

      // Add user message to chat history immediately for better UX
      const userMessage = { role: 'user', content: message };
      setChatHistory(prev => [...prev, userMessage]);
      
      // Clear input field
      setMessage('');

      // Send message to API with proper format
      const response = await axios.post('/chat', {
        session_id: sessionId,
        message: userMessage.content,
        summary_id: summaryId // Include summary_id as a fallback
      });

      // Update chat history with response from API
      if (response.data.chat_history && Array.isArray(response.data.chat_history)) {
        setChatHistory(response.data.chat_history);
      } else if (response.data.response) {
        // If chat_history is not provided, add the response as a new message
        setChatHistory(prev => [...prev, { role: 'assistant', content: response.data.response }]);
      }
    } catch (err) {
      console.error('Error sending message:', err);
      
      // Provide more detailed error message
      let errorMessage = 'Failed to send message. Please try again.';
      
      if (err.response?.status === 404) {
        errorMessage = 'Chat endpoint not found. Please check if the backend service is running.';
      } else if (err.response?.status === 400) {
        errorMessage = 'Invalid request format. Please try again with a different message.';
      } else if (err.response?.data?.detail) {
        errorMessage = err.response.data.detail;
      } else if (err.message) {
        errorMessage = `Error: ${err.message}`;
      }
      
      setError(errorMessage);
      
      // Remove the user message if there was an error
      setChatHistory(prev => prev.slice(0, -1));
    } finally {
      setLoading(false);
    }
  };

  // Handle Enter key press
  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <Paper elevation={3} sx={{ p: 3, mb: 4, height: '500px', display: 'flex', flexDirection: 'column' }}>
      <Typography variant="h5" gutterBottom sx={{ fontWeight: 600 }}>
        Chat with AI about this Summary
      </Typography>
      <Divider sx={{ mb: 2 }} />

      {/* Chat Messages */}
      <Box sx={{ flexGrow: 1, overflow: 'auto', mb: 2, p: 1 }}>
        {chatHistory.length === 0 ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
            <Typography variant="body2" color="text.secondary">
              {sessionId ? 'Start chatting about the summary...' : 'Initializing chat session...'}
            </Typography>
          </Box>
        ) : (
          chatHistory.map((msg, index) => (
            <Box 
              key={index} 
              sx={{
                display: 'flex',
                mb: 2,
                flexDirection: msg.role === 'user' ? 'row-reverse' : 'row'
              }}
            >
              <Avatar 
                sx={{ 
                  bgcolor: msg.role === 'user' ? 'primary.main' : 'secondary.main',
                  mr: msg.role === 'user' ? 0 : 1,
                  ml: msg.role === 'user' ? 1 : 0
                }}
              >
                {msg.role === 'user' ? <Person /> : <SmartToy />}
              </Avatar>
              <Paper 
                elevation={1} 
                sx={{
                  p: 2,
                  maxWidth: '70%',
                  bgcolor: msg.role === 'user' ? 'primary.light' : 'background.default',
                  borderRadius: 2
                }}
              >
                <Typography variant="body1">{msg.content}</Typography>
              </Paper>
            </Box>
          ))
        )}
        {error && (
          <Typography variant="body2" color="error" sx={{ mt: 1, textAlign: 'center' }}>
            {error}
          </Typography>
        )}
        <div ref={messagesEndRef} />
      </Box>

      {/* Message Input */}
      <Box sx={{ display: 'flex', alignItems: 'center' }}>
        <TextField
          fullWidth
          variant="outlined"
          placeholder="Type your message..."
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyPress={handleKeyPress}
          disabled={loading || !sessionId}
          multiline
          maxRows={3}
          sx={{ mr: 1 }}
        />
        <IconButton 
          color="primary" 
          onClick={handleSendMessage} 
          disabled={loading || !message.trim() || !sessionId}
          sx={{ p: 1 }}
        >
          {loading ? <CircularProgress size={24} /> : <Send />}
        </IconButton>
      </Box>
    </Paper>
  );
};

export default ChatComponent;