import numpy as np
import librosa
from scipy.signal import correlate

def calculate_time_lag(ref_audio_path, lag_audio_path, corr_sr=8000, max_shift_s=10.0, max_seconds=60.0):
    """
    Calculates the time lag with optimizations for speed.

    Args:
        ref_audio_path (str): File path to the reference audio.
        lag_audio_path (str): File path to the lagging audio.
        corr_sr (int): Sampling rate for correlation. Higher values increase accuracy but also computation time.
        max_shift_s (float): Maximum expected lag in seconds. Reduces computation by limiting the search range.
        max_seconds (float): Maximum duration of audio to consider for lag calculation.

    Returns:
        float: The time offset in seconds. A positive value indicates the
               'lagging_audio' starts after 'reference_audio'.
    """
    # Load and downsample to corr_sr for faster correlation
    ref_signal, sr_ref = librosa.load(ref_audio_path, sr=corr_sr, duration=max_seconds)
    lag_signal, sr_lag = librosa.load(lag_audio_path, sr=corr_sr, duration=max_seconds)
    print('Audio loaded and downsampled')
    
    # Ensure SRs match (should be corr_sr)
    sr = corr_sr
    
    # Limit lag_signal to avoid excessive computation
    max_shift_samples = int(max_shift_s * sr)
    lag_signal = lag_signal[:len(ref_signal) + max_shift_samples]  # Trim to reasonable length
    
    # FFT-based correlation (faster than np.correlate)
    correlation = correlate(lag_signal, ref_signal, mode='full')
    
    # Find peak within max_shift range
    zero_lag_idx = len(ref_signal) - 1
    start_idx = max(0, zero_lag_idx - max_shift_samples)
    end_idx = min(len(correlation), zero_lag_idx + max_shift_samples)
    peak_index = start_idx + np.argmax(correlation[start_idx:end_idx])
    
    offset_samples = peak_index - zero_lag_idx
    offset_seconds = offset_samples / sr
    
    return offset_seconds

# --- Usage ---
reference_file = "left.wav"
lagging_file = "lag.wav"

lag_time = calculate_time_lag(reference_file, lagging_file)
print(f"Calculated lag: {lag_time:.4f} seconds.")

# Interpretation:
# If lag_time > 0: The lagging file starts 'lag_time' seconds AFTER the reference file.
#                  To align them, you need to either trim 'lag_time' seconds from the start
#                  of the lagging file OR add 'lag_time' seconds of silence to the start
#                  of the reference file.
# If lag_time < 0: The lagging file actually starts BEFORE the reference file.