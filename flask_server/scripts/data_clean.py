import pandas as pd
import re
from textblob import TextBlob
from sentence_transformers import SentenceTransformer
from constants import qa_data_path

df = pd.read_csv(qa_data_path)

# Remove 'SPEAKER_..' (with optional timestamp, ':', '&', or whitespace) and '&' (with optional timestamp, ':', or whitespace)
pattern = r"(SPEAKER_\d+(?:\s*\([^)]+\))?[:&]?\s*|&(?:\s*\([^)]+\))?:?\s*)"
df['answer_text'] = df['answer_text'].str.replace(pattern, "", regex=True)
df['question_text'] = df['question_text'].str.replace(pattern, "", regex=True)

df.dropna(subset=['answer_text', 'question_text'], inplace=True)

# Ensure text columns are strings
df['question_text'] = df['question_text'].astype(str)
df['answer_text'] = df['answer_text'].astype(str)

# Length features
df['question_length'] = df['question_text'].str.split().str.len()
df['answer_length'] = df['answer_text'].str.split().str.len()

# Sentiment features (using TextBlob or similar)
df['question_sentiment'] = df['question_text'].apply(lambda x: TextBlob(x).sentiment.polarity)
df['answer_sentiment'] = df['answer_text'].apply(lambda x: TextBlob(x).sentiment.polarity)

# Embedding features (using SentenceTransformer)
model = SentenceTransformer('all-MiniLM-L6-v2')
df['question_embedding'] = df['question_text'].apply(lambda x: model.encode(x).tolist())
df['answer_embedding'] = df['answer_text'].apply(lambda x: model.encode(x).tolist())

print(df.head())
df.to_csv(qa_data_path, index=False)