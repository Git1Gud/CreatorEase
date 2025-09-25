import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';

const BASE_URL = 'http://localhost:8000';

const UploadPage= () => {
  const [dragActive, setDragActive] = useState<boolean>(false);
  const navigate = useNavigate();
  
  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };
  
  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (file.type.startsWith('video/')) {
        handleFile(file);
      } else {
        alert('Please upload a video file.');
      }
    }
  };
  
  const handleChange = (e) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      if (file.type.startsWith('video/')) {
        handleFile(file);
      } else {
        alert('Please upload a video file.');
      }
    }
  };
  
  const handleFile = async (file) => {
    const formData = new FormData();
    formData.append('file', file);

    try {
      // Send the file to the backend
      const response = await fetch(`${BASE_URL}/upload`, {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const result = await response.json(); // Consider defining an interface for the response
        if (result.video_id) {
          // Navigate to the edit page with video ID from backend
          navigate(`/edit?video_id=${result.video_id}`);
        } else {
          console.error('Backend did not return a video_id:', result);
          alert('File upload was successful, but could not get video ID. Please try again.');
        }
      } else {
        // Handle server errors (e.g., 500, 400)
        const errorResult = await response.json(); // Consider defining an interface for the error response
        console.error('File upload failed:', response.statusText, errorResult);
        alert(`File upload failed: ${errorResult.error || response.statusText}. Please try again.`);
      }
    } catch (error) {
      // Handle network errors or other issues with the fetch call
      console.error('Error uploading file:', error);
      alert('Error uploading file. Please check your connection and try again.');
    }
  };
  
  return (
    <div className="upload-container">
      <h2>Upload Video</h2>
      <div 
        className={`upload-area ${dragActive ? 'active' : ''}`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
      >
        <p>Drag and drop your video here</p>
        <label className="upload-button">
          <input 
            type="file" 
            accept="video/*" 
            onChange={handleChange} 
            hidden 
          />
          <span>Browse Files</span>
        </label>
      </div>
    </div>
  );
}

export default UploadPage; 