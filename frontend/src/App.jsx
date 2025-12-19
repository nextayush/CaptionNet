import React, { useState, useEffect } from 'react';
import { useDropzone } from 'react-dropzone';
import { motion, AnimatePresence } from 'framer-motion';
import { FiUploadCloud, FiImage, FiCpu, FiCheckCircle } from 'react-icons/fi';
import { generateCaption, checkServerStatus } from './api';
import './App.css'; // Import the CSS file

function App() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [caption, setCaption] = useState("");
  const [loading, setLoading] = useState(false);
  const [serverStatus, setServerStatus] = useState("checking");

  useEffect(() => {
    checkServerStatus().then((status) => {
      setServerStatus(status ? "online" : "offline");
    });
  }, []);

  const onDrop = (acceptedFiles) => {
    const file = acceptedFiles[0];
    setImage(file);
    setPreview(URL.createObjectURL(file));
    setCaption("");
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/*': [] },
    multiple: false
  });

  const handleGenerate = async () => {
    if (!image) return;
    setLoading(true);
    setCaption("");
    
    try {
      const result = await generateCaption(image);
      setCaption(result.caption);
    } catch (error) {
      alert("Error generating caption. Is the backend running?");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      
      {/* Header */}
      <motion.div 
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="header"
      >
        <h1 className="title">CaptionNet</h1>
        <p className="subtitle">Multimodal Neural Network for Image Captioning</p>
        
        <div className={`status-badge ${serverStatus === 'online' ? 'status-online' : 'status-offline'}`}>
          <div className="status-dot" style={{backgroundColor: serverStatus === 'online' ? '#10b981' : '#ef4444'}}></div>
          BACKEND: {serverStatus.toUpperCase()}
        </div>
      </motion.div>

      <div className="grid-layout">
        
        {/* Left Card: Upload */}
        <motion.div 
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          className="card"
        >
          <div 
            {...getRootProps()} 
            className={`dropzone ${isDragActive ? 'active' : ''}`}
          >
            <input {...getInputProps()} />
            {preview ? (
              <img src={preview} alt="Preview" className="preview-image" />
            ) : (
              <div className="placeholder-text">
                <FiUploadCloud className="icon-large" />
                <p>Drag & drop image here</p>
                <small>or click to select</small>
              </div>
            )}
          </div>

          <button
            onClick={handleGenerate}
            disabled={!image || loading || serverStatus === 'offline'}
            className="generate-btn"
          >
            {loading ? (
              <>
                <div className="spinner"></div>
                Analyzing Pixels...
              </>
            ) : (
              <>
                <FiCpu /> Generate Caption
              </>
            )}
          </button>
        </motion.div>

        {/* Right Card: Result */}
        <motion.div 
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          className="card"
        >
          <h3 style={{ display: 'flex', alignItems: 'center', gap: '10px', marginTop: 0 }}>
            <FiImage style={{ color: 'var(--primary)' }} /> Generated Output
          </h3>

          <div className="output-box">
            <AnimatePresence mode='wait'>
              {caption ? (
                <motion.div
                  key="result"
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0 }}
                  style={{ textAlign: 'center' }}
                >
                  <FiCheckCircle style={{ fontSize: '40px', color: '#10b981', marginBottom: '15px' }} />
                  <p className="caption-text">"{caption}"</p>
                </motion.div>
              ) : (
                <motion.div 
                  key="empty"
                  initial={{ opacity: 0 }} 
                  animate={{ opacity: 1 }}
                  style={{ color: '#9ca3af' }}
                >
                  {loading ? (
                    <p>The AI is connecting neurons...</p>
                  ) : (
                    <p>No caption generated yet.</p>
                  )}
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </motion.div>

      </div>
    </div>
  );
}

export default App;