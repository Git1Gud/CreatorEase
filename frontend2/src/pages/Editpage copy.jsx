import React, { useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { VideoEditorProvider, useVideoEditor } from '../context/VideoEditorContext';
import TranscriptionPanel from '../components/TranscriptionPanel';
import VideoPanel from '../components/VideoPanel';
import PlaybackControls from '../components/PlaybackControls';
import AudioWaveform from '../components/AudioWaveform';

const BASE_URL = 'http://localhost:8000';

const EditPageContent = () => {
  const navigate = useNavigate();
  const location = useLocation();
  // No longer need to explicitly type here if useVideoEditor is correctly typed and returns VideoEditorContextValue
  const { // Destructure directly, types come from useVideoEditor's return type
      videoRef,
      waveformRef,
      wavesurferRef, 
      regionsPluginRef, 
      videoFile,
      trimHistory,
      setVideoFile,
      setTranscription, 
      setup,
      joinRoom

      // We need a way to signal readiness from context or manage locally
      // For now, managing a local state triggered by wavesurfer events
    } = useVideoEditor();

    
//     useEffect(() => {
//   setup();
//   // joinRoom();
// }, []);

  useEffect(() => {
  const vid = videoRef.current;
  const ws = wavesurferRef.current;
  if (!vid || !ws) return;

  const handleTimeUpdate = () => {
    if (!regionsPluginRef.current) return;
    const regions = Object.values(regionsPluginRef.current.getRegions());

    for (const region of regions) {
      if (vid.currentTime >= region.start && vid.currentTime < region.end) {
        // Skip video
        vid.currentTime = region.end+0.01;

        // Sync waveform
        ws.setTime(region.end+0.01);
        break;
      }
    }
  };

  vid.addEventListener("timeupdate", handleTimeUpdate);
  return () => {
    vid.removeEventListener("timeupdate", handleTimeUpdate);
  };
}, [videoFile, trimHistory]);

  useEffect(() => {
    const params = new URLSearchParams(location.search);
    const video_id = params.get('video_id');
    
    if (video_id) {
      console.log(`Fetching video with ID: ${video_id}`);
      fetch(`${BASE_URL}/video/${video_id}`)
        .then(response => {
          if (!response.ok) {
            if (response.status === 404) {
              console.error(`Video with ID ${video_id} not found on server.`);
            } else {
              console.error(`Error fetching video ${video_id}: ${response.status} ${response.statusText}`);
            }
            throw new Error(`Server error: ${response.status}`);
          }
          return response.blob();
        })
        .then(blob => {
          const videoUrl = URL.createObjectURL(blob);
          const fetchedVideoFile = { // Uses imported VideoFile
            id: video_id,
            name: `Video ${video_id}`,
            url: videoUrl,
          };
          setVideoFile(fetchedVideoFile);
          loadDefaultTranscription();
        })
        .catch(error => {
          console.error('Error fetching video from backend or processing blob:', error);
          alert(`Could not load video: ${video_id}. Loading default video instead.`);
          loadDefaultVideo();
        });
    } else {
      loadDefaultVideo();
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [location, setVideoFile, setTranscription]); // Added setTranscription to deps as loadDefaultTranscription uses it.

  const loadDefaultTranscription = () => {
    const defaultTranscriptionPath = '/transcription.json';
    fetch(defaultTranscriptionPath)
      .then(response => {
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
      })
      .then((data) => { // Uses imported TranscriptionData
        console.log("Setting default transcription:", data); 
        setTranscription(data);
      })
      .catch(error => {
        console.error('Error loading default transcription:', error);
        setTranscription(null); 
      });
  };

  const loadDefaultVideo = () => {
    const defaultVideoPath = '/2x.mp4';
    setVideoFile({ id: 'default', name: 'Default Video (Loading...)', url: '' }); 
    fetch(defaultVideoPath)
      .then(response => response.blob())
      .then(blob => {
        const url = URL.createObjectURL(blob);
        const defaultVideo = { // Uses imported VideoFile
          id: 'default',
          name: 'Default Video',
          url: url
        };
        setVideoFile(defaultVideo);
        loadDefaultTranscription(); 
      })
      .catch(error => {
        console.error('Error loading default video:', error);
        const fallbackVideo = { // Uses imported VideoFile
          id: 'default',
          name: 'Default Video (Fallback)',
          url: 'https://storage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4'
        };
        setVideoFile(fallbackVideo);
        loadDefaultTranscription(); 
      });
  };

  if (!videoFile) {
    return <div>Loading...</div>;
  }

  return (
    <div className="edit-container">
      <h2>Edit Video: {videoFile.name}</h2>
      <button 
        className="back-button" 
        onClick={() => navigate('/')}
      >
        Back to Upload
      </button>
      
      <div className="edit-content">
        <div className="transcription-section">
          <TranscriptionPanel />
        </div>
        <VideoPanel />
      </div>
      
      <PlaybackControls />
      <AudioWaveform />
    </div>
  );
}

const EditPage = () => {
  return (
    <VideoEditorProvider>
    <EditPageContent />
</VideoEditorProvider>
  );
}

export default EditPage; 