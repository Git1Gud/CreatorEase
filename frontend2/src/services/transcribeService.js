// import { TranscriptionData, TranscriptionOptions } from '../types';

const BASE_URL =  'http://localhost:8000';

/**
 * Transcribe a Blob video by uploading to the backend.
 * @param blobUrl - URL.createObjectURL(videoBlob)
 * @param fileName - name of the file (e.g. 'video.mp4')
 * @param options - Additional transcription options from shared types
 * @returns Promise<TranscriptionData> transcription data
 */
export async function transcribeBlob(
  blobUrl, 
  fileName, 
  options = {}
) {
  const resp = await fetch(blobUrl);
  if (!resp.ok) {
    throw new Error(`Failed to fetch blob: ${resp.statusText}`);
  }
  const blob = await resp.blob();
  const formData = new FormData();
  formData.append('file', blob, fileName);
  
  if (options.model_name) formData.append('model_name', options.model_name);
  if (options.batch_size) formData.append('batch_size', options.batch_size.toString());
  if (options.language) formData.append('language', options.language);
  if (options.align_model) formData.append('align_model', options.align_model);
  if (options.highlight_words !== undefined) formData.append('highlight_words', options.highlight_words.toString());
  if (options.vad_onset !== undefined) formData.append('vad_onset', options.vad_onset.toString());
  if (options.compute_type) formData.append('compute_type', options.compute_type);

  const response = await fetch(`${BASE_URL}/transcribe/`, {
    method: 'POST',
    body: formData
  });

  if (!response.ok) {
    let errorDetail = 'Transcription failed';
    try {
      const errResponse = await response.json();
      errorDetail = errResponse.detail || errorDetail;
    } catch (e) {
      errorDetail = response.statusText || errorDetail;
    }
    throw new Error(errorDetail);
  }
  return response.json() ;
}


/**
 * Transcribe a static file already on the server.
 * @param filePath - Path relative to public root (e.g. '/video.mp4')
 * @param options - Additional transcription options from shared types
 * @returns Promise<TranscriptionData> transcription data
 */
export async function transcribeFile(
  filePath, 
  options = {}
) {
  const payload = {
    file_path: filePath,
    model_name: options.model_name ?? 'medium',
    compute_type: options.compute_type ?? 'float32',
    batch_size: options.batch_size ?? 8,
    language: options.language,
    align_model: options.align_model,
    highlight_words: options.highlight_words,
    vad_onset: options.vad_onset,
  };
  
  const response = await fetch(`${BASE_URL}/transcribe-file/`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });

  if (!response.ok) {
    let errorDetail = 'Transcription failed';
    try {
      const errResponse = await response.json();
      errorDetail = errResponse.detail || errorDetail;
    } catch (e) {
      errorDetail = response.statusText || errorDetail;
    }
    throw new Error(errorDetail);
  }
  return response.json();
}