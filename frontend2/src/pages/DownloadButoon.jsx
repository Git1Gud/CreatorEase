import React, { useState, useRef, useEffect, useCallback } from 'react';
import { useVideoEditor } from '../context/VideoEditorContext';
import socketService from '../services/socketService';

const VideoDownloadButton = () => {
  const { videoFile, trimHistory } = useVideoEditor();
  const SERVER_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
  const [processing, setProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [message, setMessage] = useState('');
  const [downloadUrl, setDownloadUrl] = useState('');
  const [autoDownloaded, setAutoDownloaded] = useState(false);
  const [streamInfo, setStreamInfo] = useState(null);

  const streamChunksRef = useRef([]);
  const receivedBytesRef = useRef(0);
  const streamInfoRef = useRef(null);
  const streamedBlobUrlRef = useRef(null);

  const panelBaseStyle = {
    padding: '24px',
    border: '1px solid var(--panel-border)',
    margin: '12px 0',
    background: 'linear-gradient(155deg, rgba(15,23,42,0.94), rgba(17,24,39,0.9))',
    borderRadius: '20px',
    boxShadow: 'var(--shadow-soft)',
    color: 'var(--text-primary)'
  };

  const progressContainerStyle = {
    width: '100%',
    background: 'rgba(15, 23, 42, 0.6)',
    borderRadius: '14px',
    height: '26px',
    position: 'relative',
    overflow: 'hidden',
    border: '1px solid rgba(148, 163, 184, 0.28)',
    boxShadow: 'inset 0 0 18px rgba(8, 15, 35, 0.6)'
  };

  const progressLabelStyle = {
    position: 'absolute',
    top: '50%',
    left: '50%',
    transform: 'translate(-50%, -50%)',
    color: 'var(--text-primary)',
    fontWeight: 600,
    fontSize: '12px',
    letterSpacing: '0.05em'
  };

  const primaryButtonStyle = {
    background: 'linear-gradient(145deg, rgba(99,102,241,0.92), rgba(129,140,248,0.82))',
    color: 'var(--text-primary)',
    padding: '12px 26px',
    border: '1px solid transparent',
    borderRadius: '14px',
    cursor: 'pointer',
    fontSize: '16px',
    fontWeight: 600,
    boxShadow: '0 16px 32px rgba(99, 102, 241, 0.28)',
    transition: 'transform var(--transition-base), box-shadow var(--transition-base)'
  };

  const secondaryButtonStyle = {
    background: 'rgba(100, 116, 139, 0.25)',
    color: 'var(--text-primary)',
    padding: '12px 26px',
    border: '1px solid rgba(148, 163, 184, 0.4)',
    borderRadius: '14px',
    cursor: 'pointer',
    fontSize: '16px',
    fontWeight: 600,
    transition: 'transform var(--transition-base), box-shadow var(--transition-base)',
    boxShadow: '0 12px 24px rgba(15, 23, 42, 0.35)'
  };

  console.log('VideoDownloadButton component rendering...');
  console.log('socketService:', socketService);
  console.log('socketService.socket:', socketService.socket);

  const decodeBase64Chunk = useCallback((base64String) => {
    const binaryString = window.atob(base64String);
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i += 1) {
      bytes[i] = binaryString.charCodeAt(i);
    }
    return bytes;
  }, []);

  const resetStreamState = useCallback(() => {
    if (streamedBlobUrlRef.current) {
      URL.revokeObjectURL(streamedBlobUrlRef.current);
      streamedBlobUrlRef.current = null;
    }
    streamChunksRef.current = [];
    receivedBytesRef.current = 0;
    streamInfoRef.current = null;
    setStreamInfo(null);
    setDownloadUrl('');
  }, []);

  // Function to trigger automatic download
  const triggerDownload = useCallback((url, filename = 'processed_video.mp4') => {
    if (!url) return;

    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    link.style.display = 'none';

    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);

    setAutoDownloaded(true);
  }, [setAutoDownloaded]);

  // Split the useEffect into multiple effects for better debugging
  useEffect(() => {
    console.log('=== VideoDownloadButton useEffect TRIGGERED ===');
    console.log('Component mounted or dependencies changed');
    console.log('socketService available:', !!socketService);
    console.log('socketService.socket available:', !!socketService.socket);

    // Check if socketService is properly initialized
    if (!socketService) {
      console.error('SocketService is not available!');
      return;
    }

    if (!socketService.socket) {
      console.warn('Socket not connected yet, waiting...');
      // Set up a timer to retry when socket is ready
      const checkSocket = setInterval(() => {
        if (socketService.socket) {
          console.log('Socket is now available, setting up listeners');
          clearInterval(checkSocket);
          setupEventListeners();
        }
      }, 1000);
      
      // Clear interval after 10 seconds to prevent infinite checking
      setTimeout(() => {
        clearInterval(checkSocket);
      }, 10000);
      
      return () => clearInterval(checkSocket);
    } else {
      setupEventListeners();
    }

    function setupEventListeners() {
      console.log('Setting up event listeners...');

      // Clear any existing listeners first
      socketService.offProcessingEvents();

      // Set up event listeners
      socketService.onProcessingStarted(({ message, progress }) => {
        console.log('VideoDownloadButton: Processing started', { message, progress });
        resetStreamState();
        setProcessing(true);
        setMessage(message);
        setProgress(progress || 0);
        setAutoDownloaded(false);
      });

      socketService.onProcessingProgress(({ message, progress }) => {
        console.log('VideoDownloadButton: Processing progress', { message, progress });
        setMessage(message);
        setProgress(prev => Math.max(prev, progress || 0));
      });

      socketService.onProcessedVideoMetadata(({ fileName, size, mimeType }) => {
        console.log('VideoDownloadButton: Received processed video metadata', { fileName, size, mimeType });
        streamChunksRef.current = [];
        receivedBytesRef.current = 0;
        streamInfoRef.current = { fileName, size, mimeType };
        setStreamInfo(streamInfoRef.current);
        setMessage(prev => prev || 'Preparing video stream...');
      });

      socketService.onProcessedVideoChunk(({ fileName, chunk, isLast }) => {
        if (chunk) {
          const bytes = decodeBase64Chunk(chunk);
          streamChunksRef.current.push(bytes);
          receivedBytesRef.current += bytes.length;

          const info = streamInfoRef.current;
          if (info?.size) {
            const fraction = Math.min(receivedBytesRef.current / info.size, 1);
            const chunkProgress = 95 + fraction * 5;
            setProgress(prev => Math.max(prev, chunkProgress));
          }
        }

        if (isLast) {
          const info = streamInfoRef.current;
          const mimeType = info?.mimeType || 'video/mp4';
          const finalFileName = info?.fileName || fileName || 'processed_video.mp4';

          try {
            const blob = new Blob(streamChunksRef.current, { type: mimeType });
            if (streamedBlobUrlRef.current) {
              URL.revokeObjectURL(streamedBlobUrlRef.current);
            }
            const objectUrl = URL.createObjectURL(blob);
            streamedBlobUrlRef.current = objectUrl;
            setDownloadUrl(objectUrl);
            setMessage('Video stream ready to download!');
            setProgress(100);

            // Trigger automatic download after a short delay to ensure state updates settle
            setTimeout(() => {
              triggerDownload(objectUrl, finalFileName);
            }, 250);
          } catch (err) {
            console.error('Failed to create blob from streamed chunks:', err);
            setMessage('Failed to assemble streamed video. Falling back to server download.');
          }
        }
      });

      socketService.onProcessingComplete(({ downloadUrl: serverDownloadUrl, message: completeMessage, fileName, size, mimeType }) => {
        console.log('VideoDownloadButton: Processing complete', { serverDownloadUrl, completeMessage });
        setProcessing(false);
        setMessage(completeMessage || 'Processing complete');
        setProgress(100);

        if (!streamedBlobUrlRef.current && serverDownloadUrl) {
          const fallbackFileName = fileName || serverDownloadUrl.split('/').pop() || 'processed_video.mp4';
          const fallbackUrl = `${SERVER_BASE_URL}${serverDownloadUrl}`;
          setDownloadUrl(fallbackUrl);
          setStreamInfo(prev => prev ?? { fileName: fallbackFileName, size, mimeType: mimeType || 'video/mp4' });
        }
      });

      socketService.onProcessingError(({ error, details }) => {
        console.log('VideoDownloadButton: Processing error', { error, details });
        setProcessing(false);
        setMessage(`Error: ${error}${details ? ` - ${details}` : ''}`);
        setProgress(0);
        resetStreamState();
        setAutoDownloaded(false);
      });

      console.log('Event listeners set up successfully');
    }

    return () => {
      console.log('VideoDownloadButton: Cleaning up event listeners');
      socketService.offProcessingEvents();
       resetStreamState();
       setProcessing(false);
       setMessage('');
       setProgress(0);
       setAutoDownloaded(false);
    };
  }, [decodeBase64Chunk, resetStreamState, triggerDownload, SERVER_BASE_URL]);

  const handleDownloadRequest = () => {
    console.log('VideoDownloadButton: Download request initiated');
    console.log('socketService.socket:', socketService.socket);
    console.log('socketService.userData:', socketService.userData);
    
    if (!videoFile || !videoFile.url) {
      alert('No video file loaded');
      return;
    }

    if (!socketService.socket) {
      alert('Socket not connected. Please refresh the page.');
      return;
    }

    if (!socketService.userData) {
      alert('User not set up. Please refresh the page.');
      return;
    }

    if (!trimHistory || trimHistory.length === 0) {
      if (!confirm('No trims to apply. The original video will be downloaded. Continue?')) {
        return;
      }
    }

    console.log('Calling socketService.processVideo...');
    socketService.processVideo(
      videoFile.url,
      trimHistory,
      videoFile.name || 'processed_video'
    );
  };

  const handleManualDownload = () => {
    if (!downloadUrl) {
      return;
    }

    const fallbackName = streamInfo?.fileName || `processed_${videoFile?.name || 'video'}_${Date.now()}.mp4`;
    triggerDownload(downloadUrl, fallbackName);
  };

  // Rest of your component remains the same...
  if (processing) {
    const progressFillStyle = {
      height: '100%',
      width: `${Math.max(0, Math.min(100, progress))}%`,
      background: 'linear-gradient(90deg, rgba(99,102,241,0.9), rgba(14,165,233,0.85))',
      borderRadius: '12px',
      transition: 'width 0.35s ease'
    };

    return (
      <div className="download-section" style={panelBaseStyle}>
        <h3 style={{ marginTop: 0, color: 'var(--text-secondary)' }}>🎬 Processing Video...</h3>
        <div style={{ marginBottom: '16px' }}>
          <div style={progressContainerStyle}>
            <div style={progressFillStyle} />
            <div style={progressLabelStyle}>{Math.round(progress)}%</div>
          </div>
        </div>
        <p style={{ color: 'var(--text-muted)', fontStyle: 'italic', marginBottom: '18px' }}>{message}</p>
        <div style={{ textAlign: 'center', marginTop: '10px', color: 'var(--accent-secondary)', letterSpacing: '0.08em' }}>
          ⚙️ Processing...
        </div>
      </div>
    );
  }

  if (downloadUrl) {
    const successPanelStyle = {
      ...panelBaseStyle,
      border: '1px solid rgba(34, 197, 94, 0.32)',
      background: 'linear-gradient(160deg, rgba(15,23,42,0.94), rgba(13,148,136,0.22))',
      boxShadow: '0 20px 40px rgba(13, 148, 136, 0.25)'
    };

    return (
      <div className="download-section" style={successPanelStyle}>
        <h3 style={{ marginTop: 0, color: 'var(--text-secondary)' }}>✅ Video Ready!</h3>
        <p style={{ color: 'var(--success)', marginBottom: '18px', fontWeight: 500 }}>
          {autoDownloaded ? '📥 Download started automatically!' : message}
        </p>

        {streamInfo && (
          <div style={{ fontSize: '13px', color: 'var(--text-muted)', marginBottom: '12px', lineHeight: 1.6 }}>
            <div><strong style={{ color: 'var(--text-secondary)' }}>File:</strong> {streamInfo.fileName}</div>
            {typeof streamInfo.size === 'number' && streamInfo.size > 0 && (
              <div><strong style={{ color: 'var(--text-secondary)' }}>Size:</strong> {(streamInfo.size / (1024 * 1024)).toFixed(2)} MB</div>
            )}
            <div><strong style={{ color: 'var(--text-secondary)' }}>Type:</strong> {streamInfo.mimeType}</div>
          </div>
        )}

        {autoDownloaded && (
          <p style={{ fontSize: '14px', color: 'var(--text-muted)', marginBottom: '12px' }}>
            If the download didn't start, click the button below:
          </p>
        )}

        <div style={{ display: 'flex', gap: '14px', justifyContent: 'center', flexWrap: 'wrap' }}>
          <button 
            onClick={handleManualDownload}
            style={{ ...primaryButtonStyle }}
          >
            📥 Download Video
          </button>
          
          <button 
            onClick={() => {
              resetStreamState();
              setAutoDownloaded(false);
              setMessage('');
              setProgress(0);
            }}
            style={{ ...secondaryButtonStyle }}
          >
            🗑️ Clear
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="download-section" style={panelBaseStyle}>
      <h3 style={{ marginTop: 0, color: 'var(--text-secondary)' }}>🎥 Download Processed Video</h3>
      <div style={{ marginBottom: '18px', lineHeight: 1.6 }}>
        <p style={{ margin: '6px 0', color: 'var(--text-primary)' }}>
          <strong style={{ color: 'var(--text-secondary)' }}>Video:</strong> {videoFile?.name || 'No video loaded'}
        </p>
        <p style={{ margin: '6px 0', color: 'var(--text-primary)' }}>
          <strong style={{ color: 'var(--text-secondary)' }}>Trims:</strong> {trimHistory?.length || 0} sections
          {trimHistory?.length > 0 && ' (will be removed from final video)'}
        </p>
        <p style={{ margin: '6px 0', fontSize: '12px', color: 'var(--text-muted)' }}>
          <strong style={{ color: 'var(--text-secondary)' }}>Debug:</strong> Socket: {socketService.socket ? '✅' : '❌'} | 
          User: {socketService.userData ? '✅' : '❌'}
        </p>
        {streamInfo && (
          <p style={{ margin: '6px 0', fontSize: '12px', color: 'var(--text-muted)' }}>
            <strong style={{ color: 'var(--text-secondary)' }}>Last stream:</strong> {streamInfo.fileName}
          </p>
        )}
      </div>
      
      <button 
        onClick={handleDownloadRequest}
        disabled={!videoFile || !videoFile.url}
        style={{
          ...primaryButtonStyle,
          width: '100%',
          opacity: videoFile?.url ? 1 : 0.4,
          cursor: videoFile?.url ? 'pointer' : 'not-allowed'
        }}
      >
        {videoFile?.url ? '🚀 Process & Download Video' : '❌ No Video Loaded'}
      </button>
      
      {message && <p style={{ color: 'var(--danger)', marginTop: '12px', textAlign: 'center' }}>{message}</p>}
    </div>
  );
};

export default VideoDownloadButton;