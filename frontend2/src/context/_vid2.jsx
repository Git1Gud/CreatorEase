import React, { createContext, useContext, useState, useRef, useCallback, ReactNode, useMemo, useEffect } from 'react';
// WaveSurfer import remains, as it's a library, not a local type
import WaveSurfer from 'wavesurfer.js'; 
import { formatTime, mergeTrimItems, detectSilencesCore, adjustSilencesWithPadding } from '../utils';
import { transcribeBlob, transcribeFile } from '../services/transcribeService';
import { io } from "socket.io-client";



const VideoEditorContext = createContext(undefined);


export function VideoEditorProvider({ children }) {
  const [videoFile, setVideoFile] = useState(null);
  const [playing, setPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [trimHistory, setTrimHistory] = useState([]);
  const [transcription, setTranscription] = useState(null);
  const [transcriptionLoading, setTranscriptionLoading] = useState(false);
  const [silenceThreshold, setSilenceThreshold] = useState(0.7);
  const [userId, setUserId] = useState(null);
    const [socketId, setSocketId] = useState(null);
    const [roomInfo, setRoomInfo] = useState(null);
    const socketRef = useRef(null);
  
  const videoRef = useRef();
  const waveformRef = useRef();
  const wavesurferRef = useRef();
  const regionsPluginRef = useRef();

  // This uses the imported TranscriptionOptions. Ensure its definition in types.ts is complete.
  const currentTranscriptionOptions = useMemo(() => ({
    model_name: 'large-v2', // Required by the imported TranscriptionOptions
    // Optional fields from imported TranscriptionOptions can be added or omitted
    align_model: 'WAV2VEC2_ASR_LARGE_LV60K_960H',
    batch_size: 4,
    compute_type: 'int8',
    highlight_words: true,
    vad_onset: 0.45
  }), []);

useEffect(() => {
  // connect to backend
  let socket = io(import.meta.env.VITE_BACKEND_URL || "http://localhost:8000");
  socketRef.current = socket;

  socket.on("connect", () => {
    console.log("Connected to server from ws:", socket.id);
    setup();
  });

  socket.on("connected", ({ userId, socketId }) => {
    console.log("Setup complete", userId, socketId);
    setUserId(userId);
    setSocketId(socketId);
  });

  socket.on("room joined", ({ roomId, leader }) => {
    console.log(`Joined room ${roomId}, leader is ${leader}`);
    setRoomInfo({ roomId, leader });
  });

  socket.on("validate", ({ newTrim }) => {
    console.log("checking for trim to be non overlapping");
    console.log(trimHistory);

    const overlap = trimHistory.some((trim) => {
      return (
        (newTrim.start >= trim.start && newTrim.start <= trim.end) || 
        (newTrim.end >= trim.start && newTrim.end <= trim.end)
      );
    });

    if (!overlap) {
      console.log("trim was indeed not overlapping");
      socket.emit("update", { newTrim, userId }); 
    } else {
      console.log("trim was overlapping");
    }
  });

  socket.on("update", ({ newTrim }) => {
    setTrimHistory(prev => mergeTrimItems(prev, [newTrim])); 
  });

  socket.on("error", (err) => {
    console.error("Server error:", err);
  });

  return () => {
    socket.disconnect();
    socketRef.current = null;
  };
}, []);

const setup = (maybeUserId) => {
  console.log("tried1");
  if (!socketRef.current) return;
  console.log("tried");
  
  socketRef.current?.emit("setup", { userId: maybeUserId });
};

const joinRoom = (roomId) => {
  console.log(userId);
  
  if (!userId) {
    setup();
    console.warn("Must call setup before joinRoom");
    return;
  }
  console.log("userid" + userId);
  
  socketRef.current?.emit("join room", { userId, roomId });
};

const updateWithValidation = (newTrim) => {
  if (roomInfo == null || !roomInfo?.roomId) {
    // Remove the setTimeout and just call joinRoom directly
    joinRoom("default-room"); // You need to provide a roomId here
    console.warn("Must call join room before validation");
    return; // Actually return here instead of continuing
  }

  console.log(roomInfo);
  console.log("camee11");
  
  // Remove the setTimeout - just emit directly
  socketRef.current?.emit("validate", {
    newTrim,
    leader: roomInfo.leader,
    room: roomInfo.roomId
  });
};
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

 const addTrimsToHistory = useCallback(() => {
  if (!wavesurferRef.current || !regionsPluginRef.current) {
    console.error("WaveSurfer or Regions plugin not initialized");
    return undefined;
  }

  const currentRegions = regionsPluginRef.current.getRegions();
  const regionsArray = Array.isArray(currentRegions) ? currentRegions : Object.values(currentRegions);

  if (regionsArray.length === 0) {
    console.log("No regions found to trim");
    return undefined;
  }

  const newTrims = regionsArray.map(region => ({
    id: region.id,
    start: region.start,
    end: region.end,
    timestamp: new Date().toISOString(),
  }));

  // Send each trim for validation
  newTrims.forEach(trim => updateWithValidation(trim));

  return newTrims; // so caller knows what got sent
}, []);


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
    // detectSilencesCore now returns SilenceRegion[] which should be compatible with TrimHistoryItem[]
    // if SilenceRegion definition in types.ts is a superset or compatible with TrimHistoryItem for these fields.
    // Let's assume detectSilencesCore returns something that can be used as TrimHistoryItem[]
    const detectedRawSilences = detectSilencesCore(transcription, duration, threshold);
    const padding = 0.2;

    // Ensure detectedRawSilences items are fully compatible with TrimHistoryItem for mergeTrimItems
    const silencesAsTrimItems = adjustSilencesWithPadding(detectedRawSilences, padding).map((s) => ({
        id: s.id || `silence-${s.start}-${s.end}-${Math.random().toString(36).substr(2, 9)}`,
        start: s.start,
        end: s.end,
        timestamp: s.timestamp || new Date().toISOString(), // Ensure timestamp
        color: s.color, // Pass along color if present
        handleStyle: s.handleStyle // Pass along handleStyle if present
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
      // Use the imported TranscriptionOptions, ensure options align
      const effectiveOptions= {
         // Start with defaults from context if any, then apply passed options
        model_name: currentTranscriptionOptions.model_name, // default from context
        align_model: currentTranscriptionOptions.align_model,
        batch_size: currentTranscriptionOptions.batch_size,
        compute_type: currentTranscriptionOptions.compute_type,
        highlight_words: currentTranscriptionOptions.highlight_words,
        vad_onset: currentTranscriptionOptions.vad_onset,
        language: currentTranscriptionOptions.language,
        ...options, // Override with any explicitly passed options
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
    userId,
    socketId,
    roomInfo,
    setup,
    joinRoom,
    updateWithValidation,
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
    transcriptionOptions: currentTranscriptionOptions, // Use the state/constant from provider
    handleZoom
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