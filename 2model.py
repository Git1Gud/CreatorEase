import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from sentence_transformers import SentenceTransformer
import ast
from sklearn.ensemble import RandomForestRegressor

# Load cleaned data
df = pd.read_csv('youtube_shorts_podcast_dataset_with_qa.csv')

# # Convert embedding columns from string to list of floats
# import ast
# df['question_embedding'] = df['question_embedding'].apply(ast.literal_eval)
# df['answer_embedding'] = df['answer_embedding'].apply(ast.literal_eval)

# # Combine embeddings (e.g., concatenate)
# question_emb = np.vstack(df['question_embedding'].apply(np.array).to_list())
# answer_emb = np.vstack(df['answer_embedding'].apply(np.array).to_list())
# X_embed = np.hstack([question_emb, answer_emb])

# Add sentiment features
# sentiment_features = df[['question_sentiment', 'answer_sentiment']].values



# Create transcript embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
df['transcript_embedding'] = df['transcript_text'].astype(str).apply(lambda x: model.encode(x).tolist())

# Use only transcript embedding and sentiment features
transcript_emb = np.vstack(df['transcript_embedding'].apply(np.array).to_list())
sentiment_features = df[['transcript_sentiment']].values

# Use only transcript embedding, sentiment, and audio features
audio_features = df[['duration', 'rms', 'zcr', 'spectral_centroid', 'spectral_bandwidth', 'tempo']].values

# X = np.hstack([transcript_emb, sentiment_features, audio_features])
X = np.hstack([transcript_emb,audio_features ])
# Target variable
target = 'views'  # Make sure your CSV has a 'views' column

# Drop rows with missing target
df = df.dropna(subset=['views'])
df['views_rating'] = pd.qcut(df['views'], 100, labels=False) + 1  # 1 to 100
y = df['views_rating'].values
# y=np.log1p(df['views'].values)
print(np.max(y), np.min(y))
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

params = {
    "n_estimators": [100, 200, 300],
    "max_depth": [4, 6, 8, 10],
    "min_samples_split": [2, 5, 10]
}

model = RandomForestRegressor(random_state=42, n_jobs=-1)
search = RandomizedSearchCV(model, params, n_iter=10, cv=3, scoring='neg_mean_squared_error')
search.fit(X_train, y_train)
print("Best params:", search.best_params_)

# Train model (Random Forest) with best parameters
model = RandomForestRegressor(**search.best_params_, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
y_pred = np.clip(np.round(y_pred), 1, 100)  # Ensure predictions are in 1-100 range
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Test MSE: {mse:.2f}")
print(f"Test R2 Score: {r2:.2f}")