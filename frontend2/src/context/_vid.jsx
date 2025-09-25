import React, { createContext, useContext, useState, useRef, useCallback, ReactNode, useMemo } from 'react';
// WaveSurfer import remains, as it's a library, not a local type
import WaveSurfer from 'wavesurfer.js'; 
import { formatTime, mergeTrimItems, detectSilencesCore, adjustSilencesWithPadding } from '../utils';
import { transcribeBlob, transcribeFile } from '../services/transcribeService';



import * as Automerge from "@automerge/automerge";
import { useDocument } from "@automerge/react";
import { Repo } from "@automerge/automerge-repo";
import { BroadcastChannelNetworkAdapter } from "@automerge/automerge-repo-network-broadcastchannel";
import { IndexedDBStorageAdapter } from "@automerge/automerge-repo-storage-indexeddb";

// Create Repo for collaboration
const repo = new Repo({
  network: [new BroadcastChannelNetworkAdapter()],
  storage: new IndexedDBStorageAdapter(),
});
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
  
  const videoRef = useRef();
  const waveformRef = useRef();
  const wavesurferRef = useRef();
  const regionsPluginRef = useRef();


  // ---- 🔹 Collaborative Trim History via Automerge ----
  const [doc, changeDoc] = useDocument(repo, "cuts-doc", () =>
    Automerge.from({ cuts: [] })
  );

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
    console.log(currentRegions);
    
    const regionsArray = Array.isArray(currentRegions) ? currentRegions : Object.values(currentRegions);

    if (regionsArray.length === 0) {
      console.log("No regions found to trim");
      return undefined;
    }
    
    changeDoc(d => {
      regionsArray.forEach(region => {
        d.cuts.push({
          id: region.id,
          start: region.start,
          end: region.end,
          timestamp: new Date().toISOString(),
        });
      });
    });

    return regionsArray;

  }, [changeDoc]);

  const removeTrimFromHistory = useCallback(
    (trimIdToRemove) => {
      changeDoc(d => {
        d.cuts = d.cuts.filter(trim => trim.id !== trimIdToRemove);
      });
    },
    [changeDoc]
  );

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
    videoFile,
    setVideoFile,
    playing,
    setPlaying,
    currentTime,
    setCurrentTime,
    duration,
    setDuration,
    trimHistory: doc?.cuts || [],
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