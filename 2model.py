import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from sentence_transformers import SentenceTransformer
import ast
from sklearn.ensemble import RandomForestRegressor
import pickle

# Load cleaned data
df = pd.read_csv('youtube_shorts_podcast_dataset_with_qa.csv')


# Create transcript embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
df['transcript_embedding'] = df['transcript_text'].astype(str).apply(lambda x: model.encode(x).tolist())

target = 'views'  # Make sure your CSV has a 'views' column

# Drop rows with missing target
df = df.dropna(subset=['views'])
df = df[(df['views'] < df['views'].quantile(0.99)) & (df['views'] > df['views'].quantile(0.01))]
df['views_rating'] = pd.qcut(df['views'], 100, labels=False) + 1  # 1 to 100
y = df['views_rating'].values

# Create features AFTER filtering
transcript_emb = np.vstack(df['transcript_embedding'].apply(np.array).to_list())
audio_features = df[['duration', 'rms', 'zcr', 'spectral_centroid', 'spectral_bandwidth', 'tempo']].values
sentiment_features = df[['transcript_sentiment','transcript_sentiment_better']].values  # <-- move this line here

X = np.hstack([transcript_emb,audio_features,sentiment_features])

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

# Save the trained model to a pickle file
with open('random_forest_views_rating_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Model saved to random_forest_views_rating_model.pkl")
