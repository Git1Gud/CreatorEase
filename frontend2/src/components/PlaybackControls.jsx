import React, { useState } from 'react';
import { useVideoEditor } from '../context/index';

const PlaybackControls = () => {
  const { // Destructure directly
    playing, 
    currentTime, 
    duration, 
    togglePlayPause, 
    formatTime, 
    addTrimsToHistory, 
    detectSilences, 
    silenceThreshold, 
    handleZoom, 
    wavesurferRef
  } = useVideoEditor();

  const [localSilenceThreshold, setLocalSilenceThreshold] = useState();

  const handleTrim = () => {
    const newTrims = addTrimsToHistory(); // Should return TrimHistoryItem[] or undefined
    if (newTrims && newTrims.length > 0) {
      console.log('Added trims to history:', newTrims);
    } else {
      console.log('No regions to trim');
    }
  };

  const handleDetectSilences = () => {
    const threshold = parseFloat(localSilenceThreshold);
    if (isNaN(threshold) || threshold < 0) {
      console.error("Invalid silence threshold value.");
      alert("Please enter a valid positive number for the silence threshold.");
      return;
    }
    detectSilences(threshold); // Should take number and return TrimHistoryItem[] or undefined
  };

  const onZoomIn = () => {
    if (wavesurferRef.current && wavesurferRef.current.options) {
      handleZoom(wavesurferRef.current.options.minPxPerSec + 20);
    }
  };

  const onZoomOut = () => {
    if (wavesurferRef.current && wavesurferRef.current.options) {
      const newZoomLevel = Math.max(1, wavesurferRef.current.options.minPxPerSec - 20);
      handleZoom(newZoomLevel);
    }
  };

  return (
    <div className="playback-controls">
      <button onClick={togglePlayPause}>
        {playing ? 'Pause' : 'Play'}
      </button>
      <button onClick={handleTrim} className="trim-button">
        Trim
      </button>
      <button onClick={onZoomIn} className="zoom-button">
        Zoom In
      </button>
      <button onClick={onZoomOut} className="zoom-button">
        Zoom Out
      </button>
      <div className="silence-detection-controls">
        <label htmlFor="silence-threshold">Silence Threshold (s):</label>
        <input 
          type="number"
          id="silence-threshold"
          name="silence-threshold"
          min="0.1" 
          step="0.1"
          value={localSilenceThreshold}
          onChange={(e) => setLocalSilenceThreshold(e.target.value)} 
          className="silence-threshold-input"
        />
        <button onClick={handleDetectSilences} className="detect-silence-button">
          Detect Silences
        </button>
      </div>
      <span className="time-display">
        {formatTime(currentTime)} / {formatTime(duration)}
      </span>
    </div>
  );
}

export default PlaybackControls; 