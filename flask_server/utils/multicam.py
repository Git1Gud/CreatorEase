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
):
    """Combine two camera videos using speaker diarization and slide-only transitions.

    - left_video_path/right_video_path: camera sources assumed time-aligned to audio.
    - audio_path: single final audio track applied to the composed video.
    - words: list of {word, start, end, speaker}.
    - output_path: final output file.
    - direction: 'ltr' or 'rtl' slide for transitions.
    - overlap: duration (s) of slide animation.
    - speaker_camera_map: optional mapping from speaker label to 'left'/'right'.
    """
    speaker_camera_map = speaker_camera_map or {}

    # Collapse to speaker-change segments
    segs = _words_to_speaker_segments(words)

    tmp_dir = os.path.join(os.path.dirname(output_path) or '.', 'tmp_multicam')
    os.makedirs(tmp_dir, exist_ok=True)

    # Prepare aligned base clips to measure size/fps
    with VideoFileClip(left_video_path) as left, VideoFileClip(right_video_path) as right:
        W, H = left.w, left.h
        fps = left.fps
        if right.w != W or right.h != H:
            right = right.resize((W, H))
        if right.fps != fps:
            right = right.set_fps(fps)

        # Effective duration is the overlap of both cameras
        video_duration = min(left.duration, right.duration)

        # If no segments (no words), default to whole video with speaker_00
        if not segs:
            mapped_segs: List[Tuple[float, float, str]] = [(0.0, video_duration, "speaker_00")]
        else:
            # Map audio segments into video timeline using offset and clamp to bounds
            mapped_segs = []
            for (start, end, spk) in segs:
                v_start = start + audio_start_offset
                v_end = end + audio_start_offset
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

            # Merge adjacent segments with same speaker (including speaker_00 + silence merging)
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

        # Build camera timeline: switch only when speaker changes; prefer not switching on very short segments
        min_switch_secs = 0.35  # avoid flicker on ultra-short bits
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
                    # extend previous segment
                    ps, pe, pc = cam_timeline[-1]
                    cam_timeline[-1] = (ps, e, pc)

        # Build per-segment subclips using the chosen camera
        piece_paths: List[str] = []
        for idx, (start, end, cam_sel) in enumerate(cam_timeline, start=1):
            cam = cam_sel if cam_sel in ('left', 'right') else _pick_camera(cam_sel, speaker_camera_map)
            src = left if cam == 'left' else right
            # Clip bounds safety
            s = max(0, min(start, src.duration - 0.001))
            e = max(0, min(end, src.duration))
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

    # Iteratively merge with slide-only transitions
    current_path = piece_paths[0]
    for next_path in piece_paths[1:]:
        merged_path = os.path.join(tmp_dir, f"merge_{uuid.uuid4().hex}.mp4")
        _slide_concat(current_path, next_path, merged_path, direction=direction, overlap=overlap)
        # cleanup previous piece files
        try:
            if os.path.exists(current_path) and os.path.basename(current_path).startswith(('piece_', 'merge_')):
                os.remove(current_path)
        except Exception:
            pass
        current_path = merged_path

    # Attach the single audio track with start offset
    with VideoFileClip(current_path) as vid, AudioFileClip(audio_path) as aud:
        # Apply audio start offset relative to video timeline
        if audio_start_offset > 0:
            aud = aud.set_start(audio_start_offset)
        elif audio_start_offset < 0:
            # If audio starts before video, trim the leading portion
            aud = aud.subclip(-audio_start_offset)
        # Trim tail if audio exceeds video
        if (aud.duration + max(0.0, audio_start_offset)) > vid.duration:
            # Ensure audio does not exceed video duration
            max_aud = max(0.0, vid.duration - max(0.0, audio_start_offset))
            aud = aud.subclip(0, max_aud)
        final = vid.set_audio(aud)
        final.write_videofile(output_path, codec='libx264', audio_codec='aac')
        try:
            final.close()
        except Exception:
            pass

    # Cleanup tmp
    try:
        if os.path.exists(current_path) and os.path.basename(current_path).startswith(('piece_', 'merge_')):
            os.remove(current_path)
        # Remove remaining temp pieces
        for p in piece_paths:
            if os.path.exists(p):
                os.remove(p)
        # Optionally remove tmp_dir if empty
        if not os.listdir(tmp_dir):
            os.rmdir(tmp_dir)
    except Exception:
        pass

    return output_path
