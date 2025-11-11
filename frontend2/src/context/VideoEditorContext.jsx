import React, { createContext, useContext, useState, useRef, useCallback, ReactNode, useMemo, useEffect } from 'react';
// WaveSurfer import remains, as it's a library, not a local type
import WaveSurfer from 'wavesurfer.js'; 
import { formatTime, mergeTrimItems, detectSilencesCore, adjustSilencesWithPadding } from '../utils';
import { transcribeBlob, transcribeFile } from '../services/transcribeService';
import socketService from '../services/socketService';

const VideoEditorContext = createContext(undefined);

export function VideoEditorProvider({ children }) {
  // Existing video editor states
  const [videoFile, setVideoFile] = useState(null);
  const [playing, setPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [trimHistory, setTrimHistory] = useState([]);
  const [transcription, setTranscription] = useState(null);
  const [transcriptionLoading, setTranscriptionLoading] = useState(false);
  const [silenceThreshold, setSilenceThreshold] = useState(0.7);
  
  // Collaboration states - now integrated into the same context
  const [isConnected, setIsConnected] = useState(false);
  const [userData, setUserData] = useState(null);
  const [roomData, setRoomData] = useState(null);
  const [isLeader, setIsLeader] = useState(false);
  const [pendingTrim, setPendingTrim] = useState(null);
  
  const videoRef = useRef();
  const waveformRef = useRef();
  const wavesurferRef = useRef();
  const regionsPluginRef = useRef();

  const currentTranscriptionOptions = useMemo(() => ({
    model_name: 'large-v2',
    align_model: 'WAV2VEC2_ASR_LARGE_LV60K_960H',
    batch_size: 4,
    compute_type: 'int8',
    highlight_words: true,
    vad_onset: 0.45
  }), []);

  // Collaboration functions
  const connectSocket = useCallback((userData = {}) => {
    try {
      socketService.connect();
      socketService.setupUser(userData);
      
      // Listen for connection confirmation
      if (socketService.socket) {
        socketService.socket.on('connected', (data) => {
          setUserData(data);
          setIsConnected(true);
          console.log('User connected with data:', data);
        });

        // Listen for room join confirmation
        socketService.socket.on('room joined', (data) => {
          setRoomData(data);
          setIsLeader(data.isLeader);
          console.log('Joined room:', data+" "+data.isLeader);
        });
      }

    } catch (error) {
      console.error('Failed to connect to socket:', error);
    }
  }, []);

  const joinRoom = useCallback((roomId = null) => {
    if (!isConnected || !userData) {
      console.error('Not connected to socket or user data not available');
      return;
    }
    
    socketService.joinRoom(roomId);
  }, [isConnected, userData]);

  const disconnect = useCallback(() => {
    socketService.disconnect();
    setIsConnected(false);
    setUserData(null);
    setRoomData(null);
    setIsLeader(false);
  }, []);

  // Socket event handlers
  useEffect(() => {
  if (!isConnected || !socketService.socket) return;

  // Handle validation requests (for leaders)
  const handleValidate = ({ newTrim }) => {
    if (!isLeader) {
      console.log('Received validation request but not leader');
      return;
    }
    
    console.log('Leader validating trim:', newTrim);
    console.log('Current trim history:', trimHistory);
    
    // Check if trim overlaps with existing trims
    const isValid = !trimHistory.some(existingTrim => {
      const overlaps = !(newTrim.end <= existingTrim.start || newTrim.start >= existingTrim.end);
      if (overlaps) {
        console.log('Overlap detected with:', existingTrim);
      }
      return overlaps;
    });

    console.log('Validation result:', isValid);

    if (isValid) {
      // If valid, add to local state first
      setTrimHistory(prev => {
        const updated = [...prev, newTrim];
        console.log('Leader updated local trim history:', updated);
        return updated;
      });
      
      // Then broadcast to room
      console.log('Broadcasting update to room');
      socketService.updateTrim(newTrim, userData.userId);
    } else {
      console.log('Trim validation failed - overlaps with existing trim');
      // Optionally send validation failed event
      if (socketService.socket) {
        socketService.socket.emit('validation failed', { 
          newTrim, 
          reason: 'overlaps with existing trim' 
        });
      }
    }
  };

  // Handle trim updates from other users
  const handleUpdate = ({ newTrim }) => {
    console.log('Received trim update:', newTrim);
    console.log('Current user ID:', userData?.userId);
    
    setTrimHistory(prev => {
      // Check if this trim already exists to avoid duplicates
      const existingTrim = prev.find(trim => trim.id === newTrim.id);
      if (existingTrim) {
        console.log('Trim already exists, skipping');
        return prev;
      }
      
      const updated = [...prev, newTrim];
      console.log('Updated trim history after receiving update:', updated);
      return updated;
    });
    
    // Clear pending trim if it matches
    setPendingTrim(prev => {
      if (prev && prev.id === newTrim.id) {
        console.log('Clearing pending trim');
        return null;
      }
      return prev;
    });
  };

  // Handle validation failures
  const handleValidationFailed = ({ newTrim, reason }) => {
    console.log('Validation failed for trim:', newTrim, 'Reason:', reason);
    setPendingTrim(null);
    // Show user feedback
    alert(`Trim validation failed: ${reason}`);
  };

  socketService.onValidate(handleValidate);
  socketService.onUpdate(handleUpdate);
  
  // Add validation failed handler
  if (socketService.socket) {
    socketService.socket.on('validation failed', handleValidationFailed);
  }

  return () => {
    socketService.offValidate();
    socketService.offUpdate();
    if (socketService.socket) {
      socketService.socket.off('validation failed');
    }
  };
}, [isConnected, isLeader, trimHistory, userData]);

  const handleZoom = useCallback((level) => {
    if (wavesurferRef.current) {
      wavesurferRef.current.zoom(level);
    }
  }, []);

  const togglePlayPause = useCallback(() => {
    if (videoRef.current) {
      if (playing) {
        videoRef.current.pause();
        wavesurferRef.current?.pause();
      } else {
        videoRef.current.play();
        wavesurferRef.current?.play();
      }
      setPlaying(prevPlaying => !prevPlaying);
    }
  }, [playing]);

  // Modified addTrimsToHistory function with collaboration support
  const addTrimsToHistory = useCallback(() => {
  if (!wavesurferRef.current || !regionsPluginRef.current) {
    console.error("WaveSurfer or Regions plugin not initialized");
    return undefined;
  }
  
  const currentRegions = regionsPluginRef.current.getRegions();
  console.log('Current regions:', currentRegions);
  
  const regionsArray = Array.isArray(currentRegions) ? currentRegions : Object.values(currentRegions);

  if (regionsArray.length === 0) {
    console.log("No regions found to trim");
    return undefined;
  }
  
  const newTrims = regionsArray.map((region) => ({
    id: region.id,
    start: region.start,
    end: region.end,
    timestamp: new Date().toISOString(),
  }));

  console.log('New trims to add:', newTrims);
  console.log('Collaboration status:', { 
    isConnected, 
    hasRoomData: !!roomData, 
    hasUserData: !!userData, 
    isLeader 
  });

  // If in collaboration mode, validate with leader first
  if (isConnected && roomData && userData) {
    newTrims.forEach(newTrim => {
      if (isLeader) {
        // If user is leader, directly add to history and broadcast
        console.log('Leader adding trim directly and broadcasting');
        setTrimHistory(prev => mergeTrimItems(prev, [newTrim]));
        socketService.updateTrim(newTrim, userData.userId);
      } else {
        // If not leader, send for validation
        console.log('Non-leader sending trim for validation:', {
          newTrim,
          leader: roomData.leader,
          room: roomData.roomId
        });
        socketService.validateTrim(newTrim, roomData.leader, roomData.roomId);
        setPendingTrim(newTrim);
      }
    });
  } else {
    // If not in collaboration mode, add directly
    console.log("Not in collaboration mode - adding directly");
    setTrimHistory(prev => mergeTrimItems(prev, newTrims));
  }

  return newTrims;
}, [isConnected, roomData, userData, isLeader]);

  const removeTrimFromHistory = useCallback((trimIdToRemove) => {
    setTrimHistory(prevTrimHistory => {
      const updatedHistory = prevTrimHistory.filter(trim => trim.id !== trimIdToRemove);
      console.log(`Removed trim ${trimIdToRemove}. New history:`, updatedHistory);
      return updatedHistory;
    });
  }, []);

  const detectSilences = useCallback((threshold) => {
    if (!transcription) {
        console.log("No transcription available to detect silences.");
        return undefined;
    }
    
    const detectedRawSilences = detectSilencesCore(transcription, duration, threshold);
    const padding = 0.2;

    const silencesAsTrimItems = adjustSilencesWithPadding(detectedRawSilences, padding).map((s) => ({
        id: s.id || `silence-${s.start}-${s.end}-${Math.random().toString(36).substr(2, 9)}`,
        start: s.start,
        end: s.end,
        timestamp: s.timestamp || new Date().toISOString(),
        color: s.color,
        handleStyle: s.handleStyle
    }));
    
    if (silencesAsTrimItems.length > 0) {
      console.log(`Detected ${silencesAsTrimItems.length} silence gaps >= ${threshold}s (after padding).`);
      setTrimHistory(prev => mergeTrimItems(prev, silencesAsTrimItems));
    } else {
      console.log(`No silence gaps found >= ${threshold}s (after padding).`);
    }
    return silencesAsTrimItems;
  }, [transcription, duration]);

  const generateTranscription = useCallback(async (file = videoFile, options = {}) => {
    const currentFile = file || videoFile;
    if (!currentFile?.url) {
      console.error('No video file to transcribe');
      return;
    }
    setTranscriptionLoading(true);
    try {
      let data;
      const effectiveOptions= {
        model_name: currentTranscriptionOptions.model_name,
        align_model: currentTranscriptionOptions.align_model,
        batch_size: currentTranscriptionOptions.batch_size,
        compute_type: currentTranscriptionOptions.compute_type,
        highlight_words: currentTranscriptionOptions.highlight_words,
        vad_onset: currentTranscriptionOptions.vad_onset,
        language: currentTranscriptionOptions.language,
        ...options,
      };
      
      if (currentFile.url.startsWith('blob:')) {
        data = await transcribeBlob(currentFile.url, currentFile.name || 'video.mp4', effectiveOptions);
      } else {
        data = await transcribeFile(currentFile.url, effectiveOptions) ;
      }
      if (data) setTranscription(data);
      else setTranscription(null);

    } catch (err) {
      console.error('Error transcribing video:', err);
      setTranscription(null);
    } finally {
      setTranscriptionLoading(false);
    }
  }, [videoFile, currentTranscriptionOptions]);

  const findWordAtTime = useCallback((time) => {
    if (!transcription || !transcription.segments) return null;
    
    for (const segment of transcription.segments) {
      if (time >= segment.start && time <= segment.end) {
        if (segment.words) {
          for (const word of segment.words) {
            if (typeof word.start === 'number' && typeof word.end === 'number' && time >= word.start && time <= word.end) {
              return word;
            }
          }
        }
        return { segment };
      }
    }
    return null;
  }, [transcription]);

  const value = {
    // Existing video editor values
    videoFile,
    setVideoFile,
    playing,
    setPlaying,
    currentTime,
    setCurrentTime,
    duration,
    setDuration,
    trimHistory,
    setTrimHistory,
    addTrimsToHistory,
    removeTrimFromHistory,
    detectSilences,
    silenceThreshold,
    setSilenceThreshold,
    transcription,
    setTranscription,
    transcriptionLoading,
    setTranscriptionLoading,
    generateTranscription,
    findWordAtTime,
    videoRef,
    waveformRef,
    wavesurferRef,
    regionsPluginRef,
    formatTime,
    togglePlayPause,
    transcriptionOptions: currentTranscriptionOptions,
    handleZoom,
    pendingTrim,
    setPendingTrim,
    
    // Collaboration values - now part of the same context
    isConnected,
    userData,
    roomData,
    isLeader,
    connectSocket,
    joinRoom,
    disconnect
  };

  return (
    <VideoEditorContext.Provider value={value}>
      {children}
    </VideoEditorContext.Provider>
  );
}

export function useVideoEditor() {
  const context = useContext(VideoEditorContext);
  if (context === undefined) {
    throw new Error('useVideoEditor must be used within a VideoEditorProvider');
  }
  return context;
}