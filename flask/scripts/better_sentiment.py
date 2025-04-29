# Run this as a separate script or before your main model code

from transformers import pipeline
import pandas as pd
from constants import qa_data_path
df = pd.read_csv(qa_data_path)

sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=0)  # device=0 for GPU

# Prepare texts (truncate to 512 chars for each)
texts = df['transcript_text'].astype(str).fillna("").apply(lambda x: x[:512]).tolist()

# Batch inference
results = sentiment_pipeline(texts, batch_size=32, truncation=True)

# Convert results to numeric sentiment
def convert_result(res):
    return res['score'] if res['label'] == 'POSITIVE' else -res['score']

df['transcript_sentiment_better'] = [convert_result(r) for r in results]
df.to_csv(qa_data_path, index=False)