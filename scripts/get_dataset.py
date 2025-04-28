import yt_dlp
import os
import csv
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
from textblob import TextBlob

def download_video(video_url, video_id, download_path="downloads/"):
    os.makedirs(download_path, exist_ok=True)
    ydl_opts = {
        'format': 'best',
        'outtmpl': os.path.join(download_path, f'{video_id}.%(ext)s'),
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
        print(f"Finished downloading: {video_url}")
    except Exception as e:
        print(f"Error downloading {video_url}: {e}")

def download_videos_from_search(results_df):
    for index, row in results_df.iterrows():
        print(index)
        video_url = f"https://www.youtube.com/watch?v={row['video_id']}"
        if video_url=='https://www.youtube.com/watch?v=5GxT0t2gsaU': continue
        print(f"Downloading video: {video_url}")
        download_video(video_url,row['video_id'])
        print(f"Finished downloading: {row['title']}")

api_key = 'AIzaSyBHh9utQCeBEFW-W2m2n2SYAV5QMGRmwM8'
youtube = build('youtube', 'v3', developerKey=api_key)

query = "sports podcast"
max_results = 5000
videos_data = []

def get_transcript_sentiment(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        full_text = " ".join([entry['text'] for entry in transcript])
        sentiment = TextBlob(full_text).sentiment.polarity
        return full_text, sentiment
    except (TranscriptsDisabled, NoTranscriptFound):
        return None, None
    except Exception as e:
        print(f"Transcript error for {video_id}: {e}")
        return None, None

def get_video_stats(video_ids):
    stats = youtube.videos().list(
        part="snippet,statistics",
        id=",".join(video_ids)
    ).execute()

    result = []
    for item in stats['items']:
        video_id = item['id']
        title = item['snippet']['title']
        channel = item['snippet']['channelTitle']
        published_at = item['snippet']['publishedAt']
        views = item['statistics'].get('viewCount', '0')
        likes = item['statistics'].get('likeCount', '0')
        comments = item['statistics'].get('commentCount', '0')

        transcript, sentiment = get_transcript_sentiment(video_id)

        result.append({
            'idx': len(videos_data),
            'video_id': video_id,
            'title': title,
            'channel': channel,
            'published_at': published_at,
            'views': views,
            'likes': likes,
            'comments': comments,
            'transcript_text': transcript if transcript else "",
            'transcript_sentiment': sentiment if sentiment is not None else ""
        })

    return result

dataset_path = "youtube_shorts_podcast_dataset.csv"

# Load existing video_ids if file exists
existing_ids = set()
if os.path.exists(dataset_path):
    with open(dataset_path, "r", encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            existing_ids.add(row['video_id'])

next_page_token = None
fieldnames = [
    'idx',
    'video_id', 'title', 'channel', 'published_at',
    'views', 'likes', 'comments', 'transcript_text', 'transcript_sentiment'
]

with open(dataset_path, "a", newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    # Write header only if file is empty
    if os.stat(dataset_path).st_size == 0:
        writer.writeheader()

    while len(videos_data) < max_results:
        print(f"Fetching videos... {len(videos_data)}")
        search_response = youtube.search().list(
            part="snippet",
            maxResults=50,
            q=query,
            type="video",
            videoDuration="short",
            order="viewCount",
            relevanceLanguage="en",
            pageToken=next_page_token
        ).execute()

        video_ids = [item['id']['videoId'] for item in search_response['items']]
        # Filter out already existing video_ids
        new_video_ids = [vid for vid in video_ids if vid not in existing_ids]
        if not new_video_ids:
            next_page_token = search_response.get('nextPageToken')
            if not next_page_token:
                break
            continue

        stats = get_video_stats(new_video_ids)
        # Filter out entries with empty transcript
        stats = [
            row for row in stats
            if row['transcript_text'].strip() != "" and
               row['transcript_sentiment'] not in [0, "", None]
        ]

        for row in stats:
            writer.writerow(row)
            existing_ids.add(row['video_id'])
            videos_data.append(row)

        next_page_token = search_response.get('nextPageToken')
        if not next_page_token:
            break

print(f"Saved {len(existing_ids)} videos to youtube_shorts_podcast_dataset.csv ✅")
