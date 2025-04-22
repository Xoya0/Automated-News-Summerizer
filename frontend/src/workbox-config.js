module.exports = {
  // Define which routes should be cached and how
  runtimeCaching: [
    {
      // Match any same-origin request that contains '/api/'
      urlPattern: new RegExp('/api/'),
      // Apply a network-first strategy
      handler: 'NetworkFirst',
      options: {
        // Name of the cache
        cacheName: 'api-cache',
        // Custom cache expiration
        expiration: {
          maxEntries: 50,
          maxAgeSeconds: 60 * 60 // 1 hour
        }
      }
    },
    {
      // Match the /chat endpoint specifically
      urlPattern: new RegExp('^\/chat$|\/chat\/.*'),
      // Use NetworkOnly for chat requests to ensure fresh data
      handler: 'NetworkOnly',
      options: {
        // Background sync for offline support
        backgroundSync: {
          name: 'chatQueue',
          options: {
            maxRetentionTime: 24 * 60 // Retry for up to 24 hours (specified in minutes)
          }
        },
        // Custom error handling
        callbacks: {
          // Return a custom response if the network request fails
          networkError: async (options) => {
            return new Response(JSON.stringify({
              error: 'Network error. Your message will be sent when you are back online.'
            }), {
              headers: { 'Content-Type': 'application/json' },
              status: 503
            });
          }
        }
      }
    },
    {
      // Match all navigation requests (HTML pages)
      urlPattern: ({ request }) => request.mode === 'navigate',
      // Use a network-first strategy
      handler: 'NetworkFirst',
      options: {
        // Name of the cache
        cacheName: 'pages-cache',
        // Custom cache expiration
        expiration: {
          maxEntries: 10,
          maxAgeSeconds: 60 * 60 * 24 // 24 hours
        }
      }
    }
  ],
  // Skip waiting and clients claim
  skipWaiting: true,
  clientsClaim: true
};