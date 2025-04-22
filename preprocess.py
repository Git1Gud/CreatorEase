import os
import pandas as pd
import librosa
import numpy as np

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

df = pd.read_csv('youtube_shorts_podcast_dataset_with_qa.csv')

for col in ['question_text', 'answer_text']:
    if col in df.columns:
        df = df.drop(columns=[col])
print(df.info())


# Ensure audio feature columns exist
audio_feature_cols = ['duration', 'rms', 'zcr', 'spectral_centroid', 'spectral_bandwidth', 'tempo']
for col in audio_feature_cols:
    if col not in df.columns:
        df[col] = np.nan

def extract_audio_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    features = {}
    features['duration'] = librosa.get_duration(y=y, sr=sr)
    features['rms'] = np.mean(librosa.feature.rms(y=y))
    features['zcr'] = np.mean(librosa.feature.zero_crossing_rate(y))
    features['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    features['spectral_bandwidth'] = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    features['tempo'] = librosa.beat.tempo(y=y, sr=sr)[0]
    return features

batch_size = 10
for idx, row in df.iterrows():
    print(idx)
    video_id = row['video_id']
    video_path = os.path.join("downloads", f"{video_id}.mp4")
    # Skip if audio features already exist
    if all(
        not pd.isna(row[col])
        for col in audio_feature_cols
    ):
        print(f"Audio features already extracted for {video_id}, skipping.")
        continue
    if not os.path.exists(video_path):
        try:
            video_path = download_video(video_id)
        except Exception as e:
            print(f"Failed to download {video_id}: {e}")
            continue

    try:
        print(f"Extracting audio features from {video_path}...")
        feats = extract_audio_features(video_path)
        for k, v in feats.items():
            df.at[idx, k] = v
        # Save after each successful processing
        if idx % batch_size == 0:
            df.to_csv('youtube_shorts_podcast_dataset_with_qa.csv', index=False)
            print('Saved intermediate results.')
    except Exception as e:
        print(f"Error processing {video_id}: {e}")
# Final save at the end
df.to_csv('youtube_shorts_podcast_dataset_with_qa.csv', index=False)

