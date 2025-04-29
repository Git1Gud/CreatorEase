import librosa
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from transformers import pipeline

def extract_audio_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    features = {}
    features['duration'] = librosa.get_duration(y=y, sr=sr)
    features['rms'] = np.mean(librosa.feature.rms(y=y))
    features['zcr'] = np.mean(librosa.feature.zero_crossing_rate(y))
    features['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    features['spectral_bandwidth'] = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    features['tempo'] = librosa.beat.tempo(y=y, sr=sr)[0]
    return np.array([
        features['duration'],
        features['rms'],
        features['zcr'],
        features['spectral_centroid'],
        features['spectral_bandwidth'],
        features['tempo']
    ])

def extract_transcript(segments):
    texts = []
    for seg in segments:
        if ':' in seg:
            text = seg.split(':', 1)[1]
            text = text.replace('||', '')
            texts.append(text.strip())
    return ' '.join(texts)

def predict_rating_for_segments(video_path, segments, model_path='random_forest_views_rating_model.pkl'):
    # Extract audio features once for the whole video
    audio_features = extract_audio_features(video_path)
    # Load models once
    emb_model = SentenceTransformer('all-MiniLM-L6-v2')
    sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    ratings = []
    for seg in segments:
        # Prepare transcript for this segment
        if ':' in seg:
            transcript_text = seg.split(':', 1)[1].replace('||', '').strip()
        else:
            transcript_text = seg.strip()
        # Transcript embedding
        transcript_emb = emb_model.encode(transcript_text)
        # Sentiment
        sentiment_result = sentiment_pipeline(transcript_text[:512])[0]
        sentiment_score = sentiment_result['score'] if sentiment_result['label'] == 'POSITIVE' else -sentiment_result['score']
        sentiment_features = np.array([0.0, sentiment_score])
        # Combine features
        X_input = np.hstack([transcript_emb, audio_features, sentiment_features]).reshape(1, -1)
        # Predict
        predicted_rating = model.predict(X_input)
        ratings.append(int(np.clip(np.round(predicted_rating[0]), 1, 100)))
    return ratings

# Example usage:
# video_path = r"D:\Self\Gen ai\subtitle test\uploads\test.mp4"
# segments = [
#     "SPEAKER_00 & SPEAKER_01 (0.11-3.21): I'd love to start with these. || years of work right there.",
#     "SPEAKER_00 & SPEAKER_01 (3.35-7.90): Someone on your team call these the real-life Tony Stark glasses. || Very hard to make each one of these.",
#     "SPEAKER_00 & SPEAKER_01 (7.92-13.74): That makes me feel incredibly optimistic. || In a world where AI gets smarter and smarter, this is probably going to be the next major platform after phones.",
#     "SPEAKER_00 & SPEAKER_01 (13.90-16.68): I miss hugging my mom. || Yeah, haptics is hard.",
#     "SPEAKER_00 & SPEAKER_01 (16.90-21.95): How does generative AI change how social media feels? || We haven't found the end yet."
# ]
# ratings = predict_rating_for_segments(video_path, segments)
# for seg, rating in zip(segments, ratings):
#     print(f"Segment: {seg}\nPredicted rating (1-100): {rating}\n")