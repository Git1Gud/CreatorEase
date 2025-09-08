import os
import uuid
import re
from typing import List, Dict, Tuple
from moviepy.editor import (
    VideoFileClip,
    CompositeVideoClip,
    concatenate_videoclips,
    AudioFileClip,
    ImageClip,
)
import numpy as np  # added
import librosa
from scipy.signal import correlate
import tempfile


def _norm_speaker(s: str) -> str:
    """Normalize diarization speaker labels to a canonical form like 'speaker_00'."""
    if not s:
        return "speaker_00"
    s = str(s).strip().lower()
    # Common forms: speaker_0, speaker-0, speaker0, speaker_00, speaker:0, speaker-01, SPEAKER_01, etc.
    m = re.search(r"(\d+)", s)
    if m:
        try:
            n = int(m.group(1))
            return f"speaker_{n:02d}"
        except Exception:
            pass
    return "speaker_00"


def _words_to_speaker_segments(words: List[Dict]) -> List[Tuple[float, float, str]]:
    """Collapse word-level timestamps with speakers into contiguous (start, end, speaker) segments.
    Assumes words are sorted by start time.
    """
    if not words:
        return []
    segs: List[Tuple[float, float, str]] = []
    first = words[0]
    current_speaker = _norm_speaker(first.get("speaker"))
    start = float(first["start"])
    last_end = float(first["end"])
    for w in words[1:]:
        spk = _norm_speaker(w.get("speaker") or current_speaker)
        w_start = float(w["start"])
        w_end = float(w["end"])
        if spk != current_speaker:
            segs.append((start, last_end, current_speaker))
            start = w_start
            current_speaker = spk
        last_end = w_end
    segs.append((start, last_end, current_speaker))
    return segs


def _pick_camera(speaker: str, mapping: Dict[str, str]) -> str:
    """Return 'left' or 'right' for a given speaker label using mapping; default to 'left'."""
    spk = _norm_speaker(speaker)
    # Normalize mapping keys
    for k, v in list(mapping.items()):
        nk = _norm_speaker(k)
        if nk != k:
            mapping.pop(k, None)
            mapping[nk] = v
    if spk in mapping:
        return mapping[spk]
    # fallback simple heuristic
    return 'left' if spk.endswith('00') or spk.endswith('0') else 'right'


def _slide_concat(a_path: str, b_path: str, out_path: str, direction: str = 'ltr', overlap: float = 0.6) -> str:
    """Create a slide/push transition by overlapping the end of A and the start of B for `overlap` seconds.
    This reduces the total duration by the overlap amount (no duration preservation). Audio is not mixed.
    """
    dir_norm = 'rtl' if str(direction).lower().startswith('r') else 'ltr'
    with VideoFileClip(a_path) as a, VideoFileClip(b_path) as b:
        W, H = a.w, a.h
        # ensure same size/fps
        if b.w != W or b.h != H:
            b = b.resize((W, H))
        if b.fps != a.fps:
            b = b.set_fps(a.fps)

        # clamp overlap to valid range
        d = max(0.0, min(overlap, a.duration, b.duration))
        if d <= 0.0:
            final = concatenate_videoclips([a.without_audio(), b.without_audio()], method="compose")
            final.write_videofile(out_path, codec='libx264', audio=False)
            try:
                final.close()
            except Exception:
                pass
            return out_path

        # parts before/after the transition
        a_head = a.subclip(0, max(0.0, a.duration - d)).without_audio()
        b_tail = b.subclip(min(d, b.duration), b.duration).without_audio()

        a_trans = a.subclip(max(0.0, a.duration - d), a.duration).without_audio()
        b_trans = b.subclip(0, min(d, b.duration)).without_audio()

        # animate positions over d seconds during the overlap
        def pos_a(t):
            p = max(0.0, min(1.0, t / d))
            if dir_norm == 'rtl':
                return (W * p, 0)  # moves right
            return (-W * p, 0)     # moves left

        def pos_b(t):
            p = max(0.0, min(1.0, t / d))
            if dir_norm == 'rtl':
                return (-W * (1 - p), 0)  # enters from left
            return (W * (1 - p), 0)       # enters from right

        transition = CompositeVideoClip([
            a_trans.set_position(pos_a),
            b_trans.set_position(pos_b),
        ], size=(W, H)).without_audio().set_duration(d)

        final = concatenate_videoclips([a_head, transition, b_tail], method="compose")
        final.write_videofile(out_path, codec='libx264', audio=False)
        try:
            final.close()
        except Exception:
            pass
    return out_path


# === Audio correlation helpers (added) ===
def _to_mono(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return x.astype(np.float32)
    return x.mean(axis=1).astype(np.float32)

def _load_wave_audio(path: str, sr: int = 8000, max_seconds: float = 120.0) -> str:
    """Ensure a WAV file exists for the reference audio and return its path.

    If input is already a .wav, returns the original path. Otherwise, writes a
    temporary PCM WAV (trimmed to max_seconds, resampled to sr) and returns that path.
    Caller is responsible for deleting the temp file when done (if path != original).
    """
    try:
        if str(path).lower().endswith('.wav'):
            return path
        with AudioFileClip(path) as a:
            if max_seconds is not None:
                max_seconds=min(max_seconds,a.duration)
                a = a.subclip(0, max_seconds)
            fd, tmp = tempfile.mkstemp(suffix='.wav')
            os.close(fd)
            wav_path = tmp
            a.write_audiofile(wav_path, fps=sr, codec='pcm_s16le', verbose=False, logger=None)
        return wav_path
    except Exception as e:
        print('error in load',e)
        return ""

def _load_wave_from_video(path: str, sr: int = 8000, max_seconds: float = 120.0) -> str:
    """Extract video audio by writing a temporary WAV and return its path; no normalization or array loading.
    
    Returns the path to the temporary WAV file. Caller is responsible for deleting it when done.
    """
    try:
        with VideoFileClip(path) as v:
            if v.audio is None:
                # If no audio, create an empty WAV file
                fd, wav_path = tempfile.mkstemp(suffix='.wav')
                os.close(fd)
                # Write a minimal silent WAV (librosa can handle empty files, but we'll create a short one)
                silent_audio = np.zeros(int(sr * 0.1), dtype=np.float32)  # 0.1 second silence
                librosa.output.write_wav(wav_path, silent_audio, sr)
                return wav_path
            a = v.audio
            if max_seconds is not None:
                max_seconds=min(max_seconds,a.duration)
                a = a.subclip(0, max_seconds)
            fd, wav_path = tempfile.mkstemp(suffix='.wav')
            os.close(fd)
            a.write_audiofile(wav_path, fps=sr, codec='pcm_s16le', verbose=False, logger=None)
        return wav_path
    except Exception as e:
        print('error in video',e)
        return ""

def calculate_time_lag(ref_audio_path, lag_audio_path, corr_sr=8000, max_shift_s=100.0, max_seconds=60.0):
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
    print(ref_audio_path,lag_audio_path,sep='/n/n')
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



def _estimate_av_offsets(reference_audio_path: str, left_video_path: str, right_video_path: str,
                         sr: int = 8000, max_shift_s: float = 100.0, max_seconds: float = 120.0) -> Tuple[float, float]:
    """Return (lag_left, lag_right) in seconds using audio cross-correlation.
    Positive lag => camera audio is delayed vs. reference audio.
    """
    ref_wav_path = _load_wave_audio(reference_audio_path, sr=sr, max_seconds=max_seconds)
    left_wav_path = _load_wave_from_video(left_video_path, sr=sr, max_seconds=max_seconds)
    right_wav_path = _load_wave_from_video(right_video_path, sr=sr, max_seconds=max_seconds)
    print("ref", ref_wav_path)
    print('left',left_wav_path)
    print('right',right_wav_path)
    lag_left = calculate_time_lag(ref_wav_path, left_wav_path, corr_sr=sr, max_shift_s=max_shift_s)
    lag_right = calculate_time_lag(ref_wav_path, right_wav_path, corr_sr=sr, max_shift_s=max_shift_s)

    # Cleanup temp WAV files
    for wav_path in [ref_wav_path, left_wav_path, right_wav_path]:
        if wav_path and wav_path != reference_audio_path and os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except Exception:
                pass

    print(lag_left, lag_right)
    return lag_left, lag_right
# === end helpers ===


def combine_multicam_with_slide(
    left_video_path: str,
    right_video_path: str,
    audio_path: str,
    words: List[Dict],
    output_path: str,
    direction: str = 'ltr',
    overlap: float = 0.6,
    speaker_camera_map: Dict[str, str] = None,
    audio_start_offset: float = 0.0,
    auto_sync: bool = True,            # added: enable correlation-based sync
    corr_sr: int = 8000,               # added: correlation sample rate
    corr_max_shift: float = 100.0,       # added: max expected offset in seconds
    corr_max_seconds: float = 120.0,   # added: only analyze up to N seconds
):
    """Combine two camera videos using speaker diarization and slide-only transitions.

    If auto_sync=True, estimate per-camera offsets to the reference audio via cross-correlation,
    align both camera timelines to audio t=0, and shift word times accordingly.
    """
    speaker_camera_map = speaker_camera_map or {}

    # 1) Estimate offsets via audio correlation (ref vs video audio)
    lag_left = 0.0
    lag_right = 0.0
    if auto_sync:
        try:
            lag_left, lag_right = _estimate_av_offsets(
                audio_path, left_video_path, right_video_path,
                sr=corr_sr, max_shift_s=corr_max_shift, max_seconds=corr_max_seconds
            )
        except Exception as e:
            import traceback
            print("Failed to estimate AV offsets:", type(e).__name__, str(e))
            traceback.print_exc()
            print('h')
            lag_left, lag_right = 0.0, 0.0
    print("l r",lag_left,lag_right)
    # Choose a common audio trim so that final video t=0 == reference audio time t0
    # Ensure both camera starts are >= 0 by picking t0 >= max(-lag_left, -lag_right, 0)

    t0 = max(0.0, -lag_left, -lag_right)

    # 2) Prepare aligned base clips
    with VideoFileClip(left_video_path) as left_raw, VideoFileClip(right_video_path) as right_raw:
        # Starting positions in sources so that their content aligns with audio at time t0
        left_start = max(0.0, t0 + lag_left)
        right_start = max(0.0, t0 + lag_right)

        left = left_raw.subclip(left_start)
        right = right_raw.subclip(right_start)

        # Match geometry/fps
        W, H = left.w, left.h
        fps = left.fps
        if right.w != W or right.h != H:
            right = right.resize((W, H))
        if right.fps != fps:
            right = right.set_fps(fps)

        # Effective duration limited by overlapped cameras
        video_duration = min(left.duration, right.duration)

        # 3) Collapse to speaker-change segments
        segs = _words_to_speaker_segments(words) if words else []

        # If no segments (no words), default to whole video with speaker_00
        if not segs:
            mapped_segs: List[Tuple[float, float, str]] = [(0.0, video_duration, "speaker_00")]
        else:
            # Map audio segments into aligned video timeline by subtracting t0 (audio trimmed by t0)
            mapped_segs = []
            for (start, end, spk) in segs:
                v_start = start - t0
                v_end = end - t0
                if v_end <= 0 or v_start >= video_duration:
                    continue
                v_start = max(0.0, v_start)
                v_end = min(video_duration, v_end)
                if v_end > v_start:
                    mapped_segs.append((v_start, v_end, _norm_speaker(spk)))

            mapped_segs.sort(key=lambda x: x[0])

            # Fill gaps with default speaker_00 (silence treated as speaker_00)
            filled: List[Tuple[float, float, str]] = []
            cur = 0.0
            for (s, e, spk) in mapped_segs:
                if s > cur:
                    filled.append((cur, s, "speaker_00"))
                filled.append((s, e, _norm_speaker(spk)))
                cur = e
            if cur < video_duration:
                filled.append((cur, video_duration, "speaker_00"))

            # Merge adjacent equal-speaker segments (including speaker_00)
            merged: List[Tuple[float, float, str]] = []
            for seg in filled:
                if not merged:
                    merged.append(seg)
                else:
                    ps, pe, pspk = merged[-1]
                    cs, ce, cspk = seg
                    if cspk == pspk and abs(cs - pe) < 1e-3:
                        merged[-1] = (ps, ce, pspk)
                    else:
                        merged.append(seg)
            mapped_segs = merged

        # 4) Build camera timeline (switch only on speaker changes; avoid flicker)
        min_switch_secs = 0.35
        cam_timeline: List[Tuple[float, float, str]] = []
        last_cam = None
        for (s, e, spk) in mapped_segs:
            desired_cam = _pick_camera(spk, speaker_camera_map)
            seg_dur = max(0.0, e - s)
            if last_cam is None:
                cam_timeline.append((s, e, desired_cam))
                last_cam = desired_cam
            else:
                if desired_cam != last_cam and seg_dur >= min_switch_secs:
                    cam_timeline.append((s, e, desired_cam))
                    last_cam = desired_cam
                else:
                    ps, pe, pc = cam_timeline[-1]
                    cam_timeline[-1] = (ps, e, pc)

        # 5) Cut subclips
        tmp_dir = os.path.join(os.path.dirname(output_path) or '.', 'tmp_multicam')
        os.makedirs(tmp_dir, exist_ok=True)

        piece_paths: List[str] = []
        for idx, (start, end, cam_sel) in enumerate(cam_timeline, start=1):
            src = left if cam_sel == 'left' else right
            s = max(0.0, min(start, src.duration - 1e-3))
            e = max(0.0, min(end, src.duration))
            if e <= s:
                continue
            sub = src.subclip(s, e).without_audio()
            piece_path = os.path.join(tmp_dir, f"piece_{idx}_{uuid.uuid4().hex}.mp4")
            sub.write_videofile(piece_path, codec='libx264', audio=False)
            try:
                sub.close()
            except Exception:
                pass
            piece_paths.append(piece_path)

    if not piece_paths:
        raise ValueError("No valid subclips were created.")

    # 6) Slide-only transitions
    current_path = piece_paths[0]
    for next_path in piece_paths[1:]:
        merged_path = os.path.join(os.path.dirname(output_path) or '.', 'tmp_multicam', f"merge_{uuid.uuid4().hex}.mp4")
        _slide_concat(current_path, next_path, merged_path, direction=direction, overlap=overlap)
        try:
            if os.path.exists(current_path) and os.path.basename(current_path).startswith(('piece_', 'merge_')):
                os.remove(current_path)
        except Exception:
            pass
        current_path = merged_path

    # 7) Attach the reference audio, trimmed by t0 (so audio starts at video t=0)
    with VideoFileClip(current_path) as vid, AudioFileClip(audio_path) as aud_raw:
        aud = aud_raw.subclip(t0) if t0 > 0 else aud_raw
        # Trim tail if audio exceeds video
        if aud.duration > vid.duration:
            aud = aud.subclip(0, vid.duration)
        final = vid.set_audio(aud)
        final.write_videofile(output_path, codec='libx264', audio_codec='aac')
        try:
            final.close()
        except Exception:
            pass

    # Cleanup temps
    try:
        if os.path.exists(current_path) and os.path.basename(current_path).startswith(('piece_', 'merge_')):
            os.remove(current_path)
        for p in piece_paths:
            if os.path.exists(p):
                os.remove(p)
        tmp_dir = os.path.join(os.path.dirname(output_path) or '.', 'tmp_multicam')
        if os.path.isdir(tmp_dir) and not os.listdir(tmp_dir):
            os.rmdir(tmp_dir)
    except Exception:
        pass

    return output_path
