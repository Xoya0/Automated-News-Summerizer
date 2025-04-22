/* eslint-disable no-restricted-globals */

// This service worker can be customized
// See https://developers.google.com/web/tools/workbox/modules/workbox-webpack-plugin

self.addEventListener('install', (event) => {
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  self.clients.claim();
});

// Custom navigation route handler for SPA
self.addEventListener('fetch', (event) => {
  // Skip non-GET requests
  if (event.request.method !== 'GET') {
    return;
  }

  const url = new URL(event.request.url);
  
  // Handle navigation requests (HTML pages)
  if (event.request.mode === 'navigate') {
    event.respondWith(
      fetch(event.request).catch(() => {
        return caches.match('/index.html');
      })
    );
    return;
  }

  // Handle API requests to /chat endpoint
  if (url.pathname.startsWith('/chat')) {
    event.respondWith(
      fetch(event.request).catch((error) => {
        console.error('Error fetching chat request:', error);
        return new Response(JSON.stringify({ error: 'Network error' }), {
          headers: { 'Content-Type': 'application/json' },
          status: 503
        });
      })
    );
    return;
  }

  // Default fetch handler
  event.respondWith(
    fetch(event.request)
      .then((response) => {
        return response;
      })
      .catch(() => {
        return caches.match(event.request);
      })
  );
});