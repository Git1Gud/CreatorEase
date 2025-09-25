
// Local type definitions for TranscriptionWord, TranscriptionSegment, TranscriptionData are removed
// as TranscriptionData is imported and it contains the structure for segments and words.

/**
 * Detect silent gaps in transcription segments exceeding threshold.
 * @param transcription - Transcription data containing segments and words with start/end times.
 * @param duration - Total duration of the media.
 * @param threshold - Minimum gap length in seconds to be considered a silence.
 * @returns Array of detected silence regions, typed as TrimHistoryItem.
 */
export function detectSilencesCore(
  transcription, 
  duration, 
  threshold
) { // Return TrimHistoryItem[]
  if (!transcription?.segments?.length || threshold <= 0) {
    return [];
  }

  const silences = []; // Store as TrimHistoryItem
  const rgba = 'rgba(255, 165, 0, 0.3)'; 
  const handleStyle = { 
    left: { backgroundColor: '#FFA500' }, 
    right: { backgroundColor: '#FFA500' } 
  };

  const firstSegment = transcription.segments[0];
  const firstWordInTranscription = firstSegment?.words?.[0];

  if (firstWordInTranscription && firstWordInTranscription.start >= threshold) {
    silences.push({
      id: `silence-0-${firstWordInTranscription.start.toFixed(3)}`,
      start: 0,
      end: firstWordInTranscription.start,
      color: rgba,
      handleStyle,
      timestamp: new Date().toISOString() // timestamp is required by TrimHistoryItem
    });
  }

  for (let i = 0; i < transcription.segments.length; i++) {
    const segment = transcription.segments[i];
    const words = segment.words || [];

    for (let j = 0; j < words.length; j++) {
      const currentWord = words[j];
      let nextWord = undefined; 

      if (j < words.length - 1) {
        nextWord = words[j + 1];
      } else if (i < transcription.segments.length - 1) {
        nextWord = transcription.segments[i + 1]?.words?.[0];
      }

      if (nextWord && typeof currentWord.end === 'number' && typeof nextWord.start === 'number') {
        const gapStart = currentWord.end;
        const gapEnd = nextWord.start;
        if (gapEnd > gapStart && (gapEnd - gapStart) >= threshold) {
          silences.push({
            id: `silence-${gapStart.toFixed(3)}-${gapEnd.toFixed(3)}`,
            start: gapStart,
            end: gapEnd,
            color: rgba,
            handleStyle,
            timestamp: new Date().toISOString() // timestamp is required
          });
        }
      }
    }
  }

  const lastSegment = transcription.segments[transcription.segments.length - 1];
  const lastWordInTranscription = lastSegment?.words?.[lastSegment.words.length - 1];

  if (lastWordInTranscription && typeof lastWordInTranscription.end === 'number' && duration - lastWordInTranscription.end >= threshold) {
    silences.push({
      id: `silence-${lastWordInTranscription.end.toFixed(3)}-${duration.toFixed(3)}`,
      start: lastWordInTranscription.end,
      end: duration,
      color: rgba,
      handleStyle,
      timestamp: new Date().toISOString() // timestamp is required
    });
  }
  return silences;
}

/**
 * Adjusts the start and end times of silence regions with a specified padding.
 * @param silences - Array of silence objects (TrimHistoryItem).
 * @param padding - The amount of padding (in seconds) to apply to start and end.
 * @param minSilenceDuration - Minimum duration for a silence to be kept after padding.
 * @returns A new array of adjusted silence regions (TrimHistoryItem).
 */
export function adjustSilencesWithPadding(
  silences, // Parameter is TrimHistoryItem[]
  padding, 
  minSilenceDuration = 0.1
) { // Return TrimHistoryItem[]
  if (!Array.isArray(silences) || typeof padding !== 'number' || padding < 0) {
    console.warn('Invalid input to adjustSilencesWithPadding, returning original silences.');
    return Array.isArray(silences) ? [...silences] : []; 
  }

  const adjustedSilences = [];
  for (const silence of silences) {
    const newStart = silence.start + padding;
    const newEnd = silence.end - padding;
    if (newStart < newEnd && (newEnd - newStart) >= minSilenceDuration) {
      adjustedSilences.push({
        ...silence, // Spreading existing TrimHistoryItem ensures all fields are present
        start: newStart,
        end: newEnd,
      });
    }
  }
  return adjustedSilences;
}