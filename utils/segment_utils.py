def group_segments_two_speakers(words_with_timestamps, min_duration=10.0, max_duration=40.0):
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

    # Post-process: iteratively merge short segments
    merged_segments = []
    idx = 0
    while idx < len(segments):
        seg = segments[idx]
        duration = seg["end"] - seg["start"]
        # If segment is short and not the last, merge with next
        while duration < min_duration and idx + 1 < len(segments):
            next_seg = segments[idx + 1]
            seg["words"] += next_seg["words"]
            seg["speakers"] = list(dict.fromkeys(seg["speakers"] + next_seg["speakers"]))
            seg["end"] = next_seg["end"]
            duration = seg["end"] - seg["start"]
            idx += 1  # Move to next for possible further merging
        merged_segments.append(seg)
        idx += 1

    # If the last segment is still short and there is more than one segment, merge with previous
    if len(merged_segments) >= 2 and (merged_segments[-1]["end"] - merged_segments[-1]["start"]) < min_duration:
        merged_segments[-2]["words"] += merged_segments[-1]["words"]
        merged_segments[-2]["speakers"] = list(dict.fromkeys(merged_segments[-2]["speakers"] + merged_segments[-1]["speakers"]))
        merged_segments[-2]["end"] = merged_segments[-1]["end"]
        merged_segments.pop(-1)

    # Post-process: split segments longer than max_duration
    final_segments = []
    for seg in merged_segments:
        duration = seg["end"] - seg["start"]
        if duration <= max_duration:
            final_segments.append(seg)
        else:
            # Split the segment into chunks of max_duration
            words = seg["words"]
            start_idx = 0
            while start_idx < len(words):
                chunk_words = []
                chunk_start = words[start_idx]["start"]
                chunk_end = chunk_start
                for j in range(start_idx, len(words)):
                    chunk_end = words[j]["end"]
                    if chunk_end - chunk_start > max_duration and j > start_idx:
                        break
                    chunk_words.append(words[j])
                # Prepare the chunk segment
                chunk_speakers = list({w.get("speaker", "unknown") for w in chunk_words})
                final_segments.append({
                    "speakers": chunk_speakers,
                    "words": chunk_words,
                    "start": chunk_start,
                    "end": chunk_end,
                    "speaker_change_idx": None  # You can refine this if needed
                })
                start_idx += len(chunk_words)

    return final_segments

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