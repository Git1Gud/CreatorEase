def group_segments_two_speakers(words_with_timestamps):
    segments = []
    if not words_with_timestamps:
        return segments

    i = 0
    n = len(words_with_timestamps)
    while i < n:
        seg_words = []
        speakers_in_seg = []
        start = words_with_timestamps[i]["start"]
        current_speaker = words_with_timestamps[i].get("speaker", "unknown")
        speakers_in_seg.append(current_speaker)
        seg_words.append(words_with_timestamps[i])
        i += 1

        while i < n and words_with_timestamps[i].get("speaker", "unknown") == current_speaker:
            seg_words.append(words_with_timestamps[i])
            i += 1

        speaker_change_idx = None
        if i < n:
            next_speaker = words_with_timestamps[i].get("speaker", "unknown")
            if next_speaker != current_speaker:
                speakers_in_seg.append(next_speaker)
                speaker_change_idx = len(seg_words)
                seg_words.append(words_with_timestamps[i])
                i += 1
                while i < n and words_with_timestamps[i].get("speaker", "unknown") == next_speaker:
                    seg_words.append(words_with_timestamps[i])
                    i += 1

        end = seg_words[-1]["end"]
        segments.append({
            "speakers": speakers_in_seg,
            "words": seg_words,
            "start": start,
            "end": end,
            "speaker_change_idx": speaker_change_idx
        })
    return segments

def print_speaker_segments(segments):
    for seg in segments:
        speakers_str = " & ".join(seg['speakers'])
        text = []
        for idx, w in enumerate(seg['words']):
            if seg.get("speaker_change_idx") is not None and idx == seg["speaker_change_idx"]:
                text.append("||")
            text.append(w['word'])
        print(f"{speakers_str} ({seg['start']:.2f}-{seg['end']:.2f}): {' '.join(text)}")

def format_speaker_segments(segments):
    formatted_segments = []
    for seg in segments:
        speakers_str = " & ".join(seg['speakers'])
        text = []
        for idx, w in enumerate(seg['words']):
            if seg.get("speaker_change_idx") is not None and idx == seg["speaker_change_idx"]:
                text.append("||")
            text.append(w['word'])
        segment_str = f"{speakers_str} ({seg['start']:.2f}-{seg['end']:.2f}): {' '.join(text)}"
        formatted_segments.append(segment_str)
    return formatted_segments