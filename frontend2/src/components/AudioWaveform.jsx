import React, { useEffect, useState } from 'react';
import { useVideoEditor } from '../context/index';
import WaveSurfer from 'wavesurfer.js';
// It's good practice to check if type definitions exist for libraries like wavesurfer.js
// and install them (e.g., npm install @types/wavesurfer.js --save-dev)
// For now, we'll use `any` or define minimal interfaces if needed.
import RegionsPlugin from 'wavesurfer.js/dist/plugins/regions.esm.js';
import MinimapPlugin from 'wavesurfer.js/dist/plugins/minimap.esm.js';


const AudioWaveform = () => {
  const { // Destructure directly, types come from useVideoEditor's return type
    videoRef,
    waveformRef,
    wavesurferRef, 
    regionsPluginRef, 
    videoFile,
    setCurrentTime,
    setDuration,
    trimHistory,
    transcription,
    findWordAtTime, 
    removeTrimFromHistory,
    // We need a way to signal readiness from context or manage locally
    // For now, managing a local state triggered by wavesurfer events
  } = useVideoEditor();

  const [isWaveSurferReady, setIsWaveSurferReady] = useState(false);

  useEffect(() => {
    const initializeWaveSurfer = () => {
      if (!videoRef.current || !waveformRef.current || !videoFile?.url || !wavesurferRef || !regionsPluginRef) return;
      
      if (wavesurferRef.current) {
        wavesurferRef.current.destroy();
        setIsWaveSurferReady(false); // Reset ready state
      }
      
      const regions = RegionsPlugin.create();
      regionsPluginRef.current = regions;
      
      const minimap = MinimapPlugin.create({
        height: 20,
        waveColor: '#ddd',
        progressColor: '#999',
        cursorColor: '#333'
      });
      
      wavesurferRef.current = WaveSurfer.create({
        container: waveformRef.current,
        waveColor: '#4F4A85',
        progressColor: '#383351',
        height: 80,
        cursorWidth: 2,
        cursorColor: '#333',
        barWidth: 2,
        barGap: 1,
        mediaControls: false,
        media: videoRef.current,
        plugins: [regions, minimap]
      });
      
      regions.enableDragSelection({
        color: 'rgba(79, 74, 133, 0.3)',
      });
      
      regions.on('region-created', (region) => { 
        // console.log('Region created:', region);
        const deleteIcon = document.createElement('span');
        deleteIcon.innerHTML = '✖';
        deleteIcon.className = 'region-delete-icon';
        (region ).element.appendChild(deleteIcon);
        deleteIcon.addEventListener('click', (event) => {
          event.stopPropagation(); 
          removeTrimFromHistory(region.id);
        });
      });
      
      regions.on('region-updated', (region) => {
        // console.log('Region updated:', region);
      });

      if (wavesurferRef.current && videoRef.current) {
        const wsCurrent = wavesurferRef.current;
        const vidCurrent = videoRef.current;

        wsCurrent.on('ready', () => {
          // console.log('WaveSurfer is ready');
          setDuration(vidCurrent.duration);
          setIsWaveSurferReady(true); // Set ready state
          
          vidCurrent.addEventListener('timeupdate', () => {
            if (wsCurrent && !vidCurrent.paused) {
                const currentVideoTime = vidCurrent.currentTime;
                if (Math.abs(currentVideoTime - wsCurrent.getCurrentTime()) > 0.1) {
                    wsCurrent.seekTo(currentVideoTime / vidCurrent.duration);
                }
                setCurrentTime(currentVideoTime);
                if (transcription && findWordAtTime) {
                    findWordAtTime(currentVideoTime);
                }
            }
          });
        });
        
        wsCurrent.on('audioprocess', (time) => {
            setCurrentTime(time);
        });

        wsCurrent.on('interaction', () => { 
          vidCurrent.currentTime = wsCurrent.getCurrentTime();
          setCurrentTime(wsCurrent.getCurrentTime());
        });
      }
    };

    // console.log(wavesurferRef.current);
    

    if (videoFile?.url && videoRef.current && waveformRef.current) {

      
      const vid = videoRef.current;
      const handleLoadedMetadata = () => {
        if(waveformRef.current){ 
            setTimeout(initializeWaveSurfer, 100);
            if(vid) setDuration(vid.duration);
        }
      };
      
      vid.addEventListener('loadedmetadata', handleLoadedMetadata);
      if (vid.readyState >= 1 && waveformRef.current) { 
           setTimeout(initializeWaveSurfer, 100);
      }
      
      return () => {
        if (vid) {
          vid.removeEventListener('loadedmetadata', handleLoadedMetadata);
        }
        if (wavesurferRef.current) { // Check before destroying
            wavesurferRef.current.destroy();
            setIsWaveSurferReady(false); // Reset ready state on cleanup
        }
      };
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps 
  }, [videoFile?.url, videoRef, waveformRef, wavesurferRef, regionsPluginRef, setDuration, setCurrentTime, transcription, findWordAtTime, removeTrimFromHistory]);

  useEffect(() => {
    if (!regionsPluginRef.current || !wavesurferRef.current || !isWaveSurferReady) return;
      
    const ws = wavesurferRef.current;
    const regions = regionsPluginRef.current;

    const handleTrimHistoryChange = () => {
      if (!regions || !ws || !wavesurferRef.current) return;

      const existingWaveSurferRegions = regions.getRegions();
      // Cast the values to Region[] as Object.values returns unknown[] by default
      const existingRegionsArray = Object.values(existingWaveSurferRegions);
      existingRegionsArray.forEach((r) => (r ).remove());
      
      trimHistory.forEach((trim) => {
        regions.addRegion({
          id: trim.id,
          start: trim.start,
          end: trim.end,
          color: trim.color || 'rgba(255, 0, 0, 0.3)',
          drag: false,
          resize: false,
        });
      });
    };

    handleTrimHistoryChange();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [trimHistory, regionsPluginRef, wavesurferRef, isWaveSurferReady]);

  return (
    <div 
      id="waveform-container" 
      ref={waveformRef} 
      style={{ 
        position: 'relative',
        height: '100px'
      }}
    >
    </div>
  );
}

export default AudioWaveform; 