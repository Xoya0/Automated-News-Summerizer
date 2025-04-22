import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';
import { register, unregister } from './serviceWorkerRegistration';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);

// Register the service worker for production
// In development, we'll use a more lenient configuration
if (process.env.NODE_ENV === 'development') {
  // Use the service worker in development mode to fix routing issues
  register({
    onUpdate: registration => {
      console.log('New content is available; please refresh.');
    },
    onSuccess: registration => {
      console.log('Content is now available offline!');
    }
  });
} else {
  // Use the service worker in production mode
  register();
}