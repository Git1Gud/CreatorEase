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

def format_speaker_segments_with_neighbors(segments, max_duration=60.0):
    """
    Returns a list where each entry is the formatted string of prev+curr+next segment merged,
    but only adds prev/next if the total duration does not exceed max_duration.
    """
    formatted_segments = []
    n = len(segments)
    for i in range(n):
        # Always include current segment
        merged_indices = [i]
        curr_words = segments[i]['words']
        curr_start = curr_words[0]['start']
        curr_end = curr_words[-1]['end']
        curr_duration = curr_end - curr_start

        # Try to add prev or next if possible (prefer the smaller one)
        prev_ok = False
        next_ok = False
        prev_duration = float('inf')
        next_duration = float('inf')

        # Check prev
        if i - 1 >= 0:
            prev_words = segments[i-1]['words']
            prev_start = prev_words[0]['start']
            prev_end = prev_words[-1]['end']
            prev_duration = curr_end - prev_start
            if prev_duration <= max_duration:
                prev_ok = True

        # Check next
        if i + 1 < n:
            next_words = segments[i+1]['words']
            next_end = next_words[-1]['end']
            next_duration = next_end - curr_start
            if next_duration <= max_duration:
                next_ok = True

        # Decide which to add
        if prev_ok and next_ok:
            # Add the one that results in the smaller duration
            if prev_duration <= next_duration:
                merged_indices = [i-1, i]
            else:
                merged_indices = [i, i+1]
        elif prev_ok:
            merged_indices = [i-1, i]
        elif next_ok:
            merged_indices = [i, i+1]
        # else: only current

        # Merge segments
        merged_speakers = []
        merged_words = []
        for j in merged_indices:
            merged_speakers.extend(segments[j]['speakers'])
            merged_words.extend(segments[j]['words'])
        merged_speakers = list(dict.fromkeys(merged_speakers))
        start = merged_words[0]['start']
        end = merged_words[-1]['end']
        text = [w['word'] for w in merged_words]
        speakers_str = " & ".join(merged_speakers)
        segment_str = f"{speakers_str} ({start:.2f}-{end:.2f}): {' '.join(text)}"
        formatted_segments.append(segment_str)
    return formatted_segments