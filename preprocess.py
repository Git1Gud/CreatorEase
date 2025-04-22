import os
import pandas as pd
from utils.transcription_utils import transcribe_audio_with_whisperx
from utils.segment_utils import group_segments_two_speakers

def download_video(video_id, output_dir="downloads"):
    import yt_dlp
    url = f"https://www.youtube.com/watch?v={video_id}"
    output_path = os.path.join(output_dir, f"{video_id}.mp4")
    if os.path.exists(output_path):
        print(f"Video already downloaded: {output_path}")
        return output_path
    os.makedirs(output_dir, exist_ok=True)
    ydl_opts = {
        'outtmpl': output_path,
        'format': 'mp4/bestvideo+bestaudio/best',
        'quiet': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return output_path

df = pd.read_csv('youtube_shorts_podcast_dataset.csv')
df.dropna(inplace=True)
print(df.info())
if os.path.exists('youtube_shorts_podcast_dataset_with_qa.csv'):
    df2=pd.read_csv('youtube_shorts_podcast_dataset_with_qa.csv')

def extract_qa_from_segments(segments):
    """
    Extracts question and answer text from segments.
    Assumes the first speaker is the question, the second is the answer.
    Returns (question_text, answer_text)
    """
    for seg in segments:
        speakers_str = " & ".join(seg['speakers'])
        text = []
        for idx, w in enumerate(seg['words']):
            if seg.get("speaker_change_idx") is not None and idx == seg["speaker_change_idx"]:
                text.append("||")
            text.append(w['word'])
        segment_str = f"{speakers_str} ({seg['start']:.2f}-{seg['end']:.2f}): {' '.join(text)}"
        # Split at '||' to separate question and answer
        if "||" in segment_str:
            parts = segment_str.split("||", 1)
            question_text = parts[0].strip()
            answer_text = parts[1].strip()
        else:
            question_text = segment_str
            answer_text = segment_str
        return question_text, answer_text

# df['question_text'] = ""
# df['answer_text'] = ""

for idx, row in df.iterrows():
    print(idx)
    video_id = row['video_id']
    video_path = os.path.join("downloads", f"{video_id}.mp4")
    if not os.path.exists(video_path):
        try:
            video_path = download_video(video_id)
        except Exception as e:
            print(f"Failed to download {video_id}: {e}")
            continue
    # Check if this video has already been processed in df2
    if 'df2' in locals() and not df2.empty:
        if video_id in df2['video_id'].values:
            print("Already processed")
            continue
    try:
        print(f"Transcribing {video_path}...")
        words_with_timestamps = transcribe_audio_with_whisperx(video_path, model_name="small", device="cuda")
        segments = group_segments_two_speakers(words_with_timestamps)
        question_text, answer_text = extract_qa_from_segments(segments)
        print(question_text, answer_text)
        df.at[idx, 'question_text'] = question_text
        df.at[idx, 'answer_text'] = answer_text
        # Save after each successful processing
        df.to_csv('youtube_shorts_podcast_dataset_with_qa.csv', index=False)
    except Exception as e:
        print(f"Error processing {video_id}: {e}")

