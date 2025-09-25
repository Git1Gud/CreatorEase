import os
import uuid
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Tuple, Dict

import numpy as np
import librosa
from scipy.signal import correlate
from moviepy.editor import (
	VideoFileClip,
	AudioFileClip,
	clips_array,
)


FFMPEG_BIN = os.environ.get("FFMPEG_BIN", "ffmpeg")


class VideoSyncError(Exception):
	"""Raised when the sync pipeline cannot complete successfully."""


@dataclass
class SyncResult:
	output_path: str
	target_fps: float
	offsets: Dict[str, float]
	t0: float


def _run_ffmpeg(args, *, check: bool = True):
	"""Run an ffmpeg command with basic logging."""
	cmd = [FFMPEG_BIN] + args
	process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
	if check and process.returncode != 0:
		raise VideoSyncError(f"ffmpeg command failed ({' '.join(cmd)}): {process.stderr.strip()}")
	return process


def _convert_to_cfr(input_path: str, output_dir: str, target_fps: float) -> str:
	"""Convert a video file to constant frame rate using ffmpeg."""
	base = os.path.splitext(os.path.basename(input_path))[0]
	output_path = os.path.join(output_dir, f"{base}_cfr_{int(target_fps)}fps.mp4")
	if os.path.abspath(input_path) == os.path.abspath(output_path):
		output_path = os.path.join(output_dir, f"{uuid.uuid4().hex}_cfr.mp4")

	_run_ffmpeg([
		"-i", input_path,
		"-r", str(target_fps),
		"-vsync", "cfr",
		"-y", output_path,
	])
	return output_path


def _extract_audio(input_path: str, output_dir: str, sr: int = 16000) -> str:
	"""Extract mono WAV audio from a media file."""
	base = os.path.splitext(os.path.basename(input_path))[0]
	output_path = os.path.join(output_dir, f"{base}_{sr}hz.wav")
	_run_ffmpeg([
		"-i", input_path,
		"-ac", "1",
		"-ar", str(sr),
		"-vn",
		"-y", output_path,
	])
	return output_path


def calculate_time_lag(ref_audio_path: str, lag_audio_path: str, corr_sr: int = 8000,
					   max_shift_s: float = 10.0, max_seconds: float = 60.0) -> float:
	"""FFT-based correlation to estimate lag between reference and lagging audio."""
	ref_signal, _ = librosa.load(ref_audio_path, sr=corr_sr, duration=max_seconds)
	lag_signal, _ = librosa.load(lag_audio_path, sr=corr_sr, duration=max_seconds)

	if len(ref_signal) == 0 or len(lag_signal) == 0:
		raise VideoSyncError("Unable to load audio samples for correlation.")

	max_shift_samples = int(max_shift_s * corr_sr)
	lag_signal = lag_signal[:len(ref_signal) + max_shift_samples]

	correlation = correlate(lag_signal, ref_signal, mode="full")
	zero_lag_idx = len(ref_signal) - 1
	start_idx = max(0, zero_lag_idx - max_shift_samples)
	end_idx = min(len(correlation), zero_lag_idx + max_shift_samples)

	restricted = correlation[start_idx:end_idx]
	if restricted.size == 0:
		raise VideoSyncError("Correlation window is empty.")

	peak_index = start_idx + int(np.argmax(restricted))
	offset_samples = peak_index - zero_lag_idx
	return offset_samples / corr_sr


def _prepare_inputs(video1_path: str, video2_path: str, audio_path: str, target_fps: float,
					corr_sr: int, temp_dir: str):
	"""Convert videos to CFR and extract audio references."""
	video1_cfr = _convert_to_cfr(video1_path, temp_dir, target_fps)
	video2_cfr = _convert_to_cfr(video2_path, temp_dir, target_fps)

	audio_ref = _extract_audio(audio_path, temp_dir, sr=corr_sr)
	video1_audio = _extract_audio(video1_cfr, temp_dir, sr=corr_sr)
	video2_audio = _extract_audio(video2_cfr, temp_dir, sr=corr_sr)

	return video1_cfr, video2_cfr, audio_ref, video1_audio, video2_audio


def _align_and_stitch(video1_path: str, video2_path: str, audio_path: str,
					  offsets: Tuple[float, float], output_path: str,
					  target_fps: float) -> SyncResult:
	"""Build a side-by-side composite synchronized to the reference audio."""
	lag_v1, lag_v2 = offsets

	with VideoFileClip(video1_path) as v1, VideoFileClip(video2_path) as v2, AudioFileClip(audio_path) as ref_audio:
		if v1.duration is None or v2.duration is None or ref_audio.duration is None:
			raise VideoSyncError("Unable to determine clip durations for stitching.")

		t0 = max(0.0, -lag_v1, -lag_v2)
		v1_start = max(0.0, t0 + lag_v1)
		v2_start = max(0.0, t0 + lag_v2)

		max_duration = min(
			v1.duration - v1_start,
			v2.duration - v2_start,
			ref_audio.duration - t0,
		)

		if max_duration <= 0:
			raise VideoSyncError("No overlapping duration after alignment. Check offsets.")

		v1_clip = v1.subclip(v1_start, v1_start + max_duration)
		v2_clip = v2.subclip(v2_start, v2_start + max_duration)
		audio_clip = ref_audio.subclip(t0, t0 + max_duration)

		target_height = min(v1_clip.h, v2_clip.h)
		v1_resized = v1_clip.resize(height=target_height)
		v2_resized = v2_clip.resize(height=target_height)

		final_clip = clips_array([[v1_resized, v2_resized]])
		final_clip = final_clip.set_audio(audio_clip).set_fps(target_fps)

		os.makedirs(os.path.dirname(output_path), exist_ok=True)
		final_clip.write_videofile(
			output_path,
			codec="libx264",
			audio_codec="aac",
			fps=target_fps,
			verbose=False,
			logger=None,
		)

		final_clip.close()
		v1_clip.close()
		v2_clip.close()
		audio_clip.close()

	return SyncResult(
		output_path=output_path,
		target_fps=target_fps,
		offsets={"video1": lag_v1, "video2": lag_v2},
		t0=t0,
	)


def sync_videos_with_reference_audio(video1_path: str, video2_path: str, audio_path: str,
									 output_dir: str, target_fps: float = 30.0,
									 corr_sr: int = 8000, max_shift_s: float = 10.0,
									 max_seconds: float = 60.0) -> SyncResult:
	"""Full pipeline: CFR conversion, correlation, and stitched output."""
	if not os.path.exists(video1_path) or not os.path.exists(video2_path) or not os.path.exists(audio_path):
		raise VideoSyncError("One or more input files do not exist.")

	os.makedirs(output_dir, exist_ok=True)

	with tempfile.TemporaryDirectory(prefix="video_sync_") as tmpdir:
		video1_cfr, video2_cfr, audio_ref, video1_audio, video2_audio = _prepare_inputs(
			video1_path, video2_path, audio_path, target_fps, corr_sr, tmpdir
		)

		lag_v1 = calculate_time_lag(audio_ref, video1_audio, corr_sr=corr_sr,
									 max_shift_s=max_shift_s, max_seconds=max_seconds)
		lag_v2 = calculate_time_lag(audio_ref, video2_audio, corr_sr=corr_sr,
									 max_shift_s=max_shift_s, max_seconds=max_seconds)

		output_path = os.path.join(output_dir, f"synced_{uuid.uuid4().hex}.mp4")

		result = _align_and_stitch(
			video1_cfr,
			video2_cfr,
			audio_path,
			(lag_v1, lag_v2),
			output_path,
			target_fps,
		)

	return result

