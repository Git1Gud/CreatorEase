import React from 'react';
import { useVideoEditor } from '../context/VideoEditorContext';

// Removed local placeholder for VideoFile

const VideoPanel= () => {
  const { 
    videoFile, 
    videoRef, 
    setPlaying, 
  } = useVideoEditor(); // Removed explicit type annotation to allow inference

  if (!videoFile) {
    // You might want a more styled loading state or specific message
    return <div className="video-panel-loading">Loading video...</div>;
  }

  return (
    <div className="video-panel">
      {/* Consider making the title dynamic or removing if not essential */}
      <h3>Video Preview: {videoFile.name}</h3> 
      <video 
        ref={videoRef}
        src={videoFile.url || null} 
        controls // It's good practice to add controls to the video element for accessibility and usability
        className="video-preview"
        onPlay={() => setPlaying(true)}
        onPause={() => setPlaying(false)}
        // You might want to handle other video events like onLoadedMetadata, onError, etc.
      />
    </div>
  );
}

export default VideoPanel; 