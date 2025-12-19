import axios from 'axios';

// 1. Define Backend URL
// Ensure this matches the port your FastAPI is running on (usually 8000)
const API_URL = 'http://localhost:8000';

/**
 * Pings the backend to check if it's online.
 * Used to update the status badge in the UI.
 */
export const checkServerStatus = async () => {
  try {
    const response = await axios.get(`${API_URL}/`);
    // If we get a 200 OK, the server is online
    return response.status === 200;
  } catch (error) {
    console.error("Server connection failed:", error);
    return false;
  }
};

/**
 * Sends an image file to the backend for captioning.
 * * @param {File} imageFile - The image object from the dropzone
 * @param {string} strategy - 'beam' or 'greedy' (optional)
 * @returns {Promise<Object>} - The JSON response { caption: "..." }
 */
export const generateCaption = async (imageFile, strategy = 'beam') => {
  // We must use FormData to send files via HTTP
  const formData = new FormData();
  formData.append('file', imageFile);
  
  try {
    const response = await axios.post(`${API_URL}/predict?strategy=${strategy}`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  } catch (error) {
    console.error("Error generating caption:", error);
    throw error; // Rethrow so the UI can show an error alert
  }
};