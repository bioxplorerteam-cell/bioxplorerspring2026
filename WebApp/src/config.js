// Set REACT_APP_API_URL in a .env file (or Vercel environment variables) to your public API URL.
// Falls back to localhost for local development.
const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5001';

export default API_URL;
