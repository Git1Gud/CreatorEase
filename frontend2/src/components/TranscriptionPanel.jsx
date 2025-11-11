import React, { useState, useCallback } from 'react';
import { useVideoEditor } from '../context/VideoEditorContext';

const TranscriptionPanel = () => {
  const {
    transcription,
    transcriptionLoading,
    regionsPluginRef,
    currentTime,
    setCurrentTime,
    trimHistory
  } = useVideoEditor();

  const [error] = useState(null);
  const [selectedWords, setSelectedWords] = useState(new Set());
  const [isSelecting, setIsSelecting] = useState(false);

  // Function to add word/sentence selection as a region on waveform
  const addSelectionAsRegion = useCallback((startTime, endTime, source = 'transcription') => {
    if (!regionsPluginRef.current) {
      console.warn('Regions plugin not available');
      return;
    }

    try {
      // Create a new region on the waveform
      const regionId = `transcription-region-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
      
      const newRegion = regionsPluginRef.current.addRegion({
        id: regionId,
        start: startTime,
        end: endTime,
        color: source === 'word' ? 'rgba(255, 193, 7, 0.3)' : 'rgba(33, 150, 243, 0.3)', // Different colors for words vs sentences
        drag: true,
        resize: true,
        attributes: {
          source: source,
          createdFrom: 'transcription'
        }
      });

      console.log(`Added ${source} region:`, { start: startTime, end: endTime, id: regionId });

      // Optionally seek to the region start
      if (setCurrentTime) {
        setCurrentTime(startTime);
      }

      return newRegion;
    } catch (error) {
      console.error('Error adding transcription region:', error);
    }
  }, [regionsPluginRef, setCurrentTime]);

  // Handle single word click
  const handleWordClick = useCallback((word) => {
    if (!word.start || !word.end) return;

    // If we're in selection mode, toggle word selection
    if (isSelecting) {
      setSelectedWords(prev => {
        const newSelection = new Set(prev);
        const wordId = `${word.start}-${word.end}`;
        if (newSelection.has(wordId)) {
          newSelection.delete(wordId);
        } else {
          newSelection.add(wordId);
        }
        return newSelection;
      });
    } else {
      // Single word region
      addSelectionAsRegion(word.start, word.end, 'word');
    }

    // Seek to word position
    if (setCurrentTime) {
      setCurrentTime(word.start);
    }
  }, [isSelecting, addSelectionAsRegion, setCurrentTime]);

  // Handle sentence click (click on segment)
  const handleSentenceClick = useCallback((segment) => {
    if (!segment.start || !segment.end) return;

    addSelectionAsRegion(segment.start, segment.end, 'sentence');

    // Seek to sentence start
    if (setCurrentTime) {
      setCurrentTime(segment.start);
    }
  }, [addSelectionAsRegion, setCurrentTime]);

  // Process selected words into regions
  const processSelectedWords = useCallback(() => {
    if (selectedWords.size === 0) return;

    // Collect all selected word times
    const selectedTimes = [];
    transcription.segments.forEach(segment => {
      if (segment.words) {
        segment.words.forEach(word => {
          const wordId = `${word.start}-${word.end}`;
          if (selectedWords.has(wordId)) {
            selectedTimes.push({ start: word.start, end: word.end, word: word.word });
          }
        });
      }
    });

    if (selectedTimes.length === 0) return;

    // Sort by start time
    selectedTimes.sort((a, b) => a.start - b.start);

    // Group consecutive words into ranges
    const ranges = [];
    let currentRange = { 
      start: selectedTimes[0].start, 
      end: selectedTimes[0].end,
      words: [selectedTimes[0].word]
    };

    for (let i = 1; i < selectedTimes.length; i++) {
      const current = selectedTimes[i];
      const gap = current.start - currentRange.end;

      // If gap is small (less than 0.5 seconds), extend current range
      if (gap < 0.5) {
        currentRange.end = current.end;
        currentRange.words.push(current.word);
      } else {
        // Start new range
        ranges.push(currentRange);
        currentRange = { 
          start: current.start, 
          end: current.end,
          words: [current.word]
        };
      }
    }
    ranges.push(currentRange);

    // Create regions for each range
    ranges.forEach((range, index) => {
      addSelectionAsRegion(range.start, range.end, `multi-word-${range.words.length}`);
    });

    // Clear selection
    setSelectedWords(new Set());
    setIsSelecting(false);

    console.log(`Created ${ranges.length} region(s) from ${selectedWords.size} selected words`);
  }, [selectedWords, transcription, addSelectionAsRegion]);

  // Check if word is currently being trimmed
  const isWordTrimmed = useCallback((word) => {
    if (!word.start || !word.end) return false;
    return trimHistory.some(trim => 
      word.start >= trim.start && word.end <= trim.end
    );
  }, [trimHistory]);

  // Check if word is currently in a region
  const isWordInRegion = useCallback((word) => {
    if (!word.start || !word.end || !regionsPluginRef.current) return false;
    
    const regions = regionsPluginRef.current.getRegions();
    return regions.some(region => 
      word.start >= region.start && word.end <= region.end
    );
  }, [regionsPluginRef]);

  if (error) {
    return (
      <div className="transcription-panel">
        <h3>Transcription</h3>
        <div className="transcription-error">
          <p>Error: {error}</p>
          <p>Please check the console for more details.</p>
        </div>
      </div>
    );
  }
  
  if (transcriptionLoading) {
    return (
      <div className="transcription-panel">
        <h3>Transcription</h3>
        <div className="transcription-loading">
          <p>Generating transcription with WhisperX...</p>
          <p>This may take a few minutes.</p>
        </div>
      </div>
    );
  }
  
  if (!transcription || !transcription.segments) {
    return (
      <div className="transcription-panel">
        <h3>Transcription</h3>
        <div className="transcription-content">
          <p>No transcription available.</p>
        </div>
      </div>
    );
  }
  
  return (
    <div className="transcription-panel">
      <div className="transcription-header">
        <h3>Transcription</h3>
        <div className="summary-color-guide">
          <span className="summary-badge">
            <span className="summary-pill summary-pill--current">Current</span>
          </span>
          <span className="summary-badge">
            <span className="summary-pill summary-pill--selected">Selected</span>
          </span>
          <span className="summary-badge">
            <span className="summary-pill summary-pill--region">Region</span>
          </span>
          <span className="summary-badge">
            <span className="summary-pill summary-pill--trimmed">Trimmed</span>
          </span>
        </div>
      </div>
      
      {/* Controls */}
      <div className="transcription-controls">
        <div className="transcription-controls__actions">
          <button
            type="button"
            onClick={() => setIsSelecting(!isSelecting)}
            className={`control-button ${isSelecting ? 'control-button--danger' : 'control-button--accent'}`}
          >
            {isSelecting ? 'Cancel Selection' : 'Select Multiple Words'}
          </button>
          
          {isSelecting && selectedWords.size > 0 && (
            <>
              <button
                type="button"
                onClick={processSelectedWords}
                className="control-button control-button--success"
              >
                Create Region from {selectedWords.size} Selected Word{selectedWords.size > 1 ? 's' : ''}
              </button>
              
              <button
                type="button"
                onClick={() => setSelectedWords(new Set())}
                className="control-button control-button--warning"
              >
                Clear Selection
              </button>
            </>
          )}
        </div>
        
        <div className="transcription-controls__hints">
          <p className="transcription-controls__hint">
            <strong>Word:</strong> Create region | <strong>Sentence:</strong> Create region for entire sentence
          </p>
          {isSelecting && (
            <p className="transcription-controls__hint">
              💡 Click words to select, then "Create Region"
            </p>
          )}
        </div>
      </div>

      {/* Transcription Content */}
      <div className="transcription-content">
        {transcription.segments.map((segment) => (
          <div key={segment.id} className="transcript-segment">
            {/* Segment timestamp and sentence-level controls */}
            <div
              className="segment-meta"
            onClick={() => handleSentenceClick(segment)}
            >
              <span className="segment-timestamp">
                {Math.floor(segment.start / 60)}:{(segment.start % 60).toFixed(1).padStart(4, '0')} - {Math.floor(segment.end / 60)}:{(segment.end % 60).toFixed(1).padStart(4, '0')}
              </span>
              <button
                type="button"
                className="segment-action-button"
                title="Click to create a region for this entire sentence"
              >
                Create Region
              </button>
            </div>

            {/* Word-level display */}
            <div className="segment-words">
              {segment.words ? (
                segment.words.map((word, wordIndex) => {
                  const wordId = `${word.start}-${word.end}`;
                  const isSelected = selectedWords.has(wordId);
                  const isTrimmed = isWordTrimmed(word);
                  const isInRegion = isWordInRegion(word);
                  const isCurrentWord = currentTime >= word.start && currentTime <= word.end;

                  const wordClasses = ['transcript-word'];

                  if (isCurrentWord) {
                    wordClasses.push('current');
                  } else if (isTrimmed) {
                    wordClasses.push('trimmed');
                  } else {
                    if (isInRegion) {
                      wordClasses.push('in-region');
                    }
                    if (isSelected) {
                      wordClasses.push('selected');
                    }
                  }

                  return (
                    <span
                      key={`${segment.id}-word-${wordIndex}`}
                      onClick={() => handleWordClick(word)}
                      className={wordClasses.join(' ')}
                      title={`${word.word} (${word.start.toFixed(1)}s - ${word.end.toFixed(1)}s)${isInRegion ? ' - In Region' : ''}`}
                    >
                      {word.word}
                    </span>
                  );
                })
              ) : (
                // Fallback if no word-level data
                <span
                  onClick={() => handleSentenceClick(segment)}
                  className="segment-fallback-text"
                >
                  {segment.text}
                </span>
              )}
            </div>
          </div>
        ))}
      </div>

      {trimHistory.length > 0 && (
        <div className="active-trims-banner">
          <strong>Active Trims:</strong> {trimHistory.length} section{trimHistory.length !== 1 ? 's' : ''} will be removed from the video
        </div>
      )}
    </div>
  );
}

export default TranscriptionPanel;