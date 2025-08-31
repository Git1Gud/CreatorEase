import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sentence_transformers import SentenceTransformer
import pickle
import matplotlib.pyplot as plt
import os
from constants import qa_data_path, model_path, image_path


# Load cleaned data
df = pd.read_csv(qa_data_path)

# Create transcript embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
df['transcript_embedding'] = df['transcript_text'].astype(str).apply(lambda x: model.encode(x).tolist())

# Filter and prepare target
df = df.dropna(subset=['views'])
df = df[(df['views'] < df['views'].quantile(0.99)) & (df['views'] > df['views'].quantile(0.01))]
df['views_rating'] = pd.qcut(df['views'], 100, labels=False) + 1  # 1 to 100
y = df['views_rating'].values

# Create features AFTER filtering
transcript_emb = np.vstack(df['transcript_embedding'].apply(np.array).to_list())
audio_features = df[['duration', 'rms', 'zcr', 'spectral_centroid', 'spectral_bandwidth', 'tempo']].values
sentiment_features = df[['transcript_sentiment', 'transcript_sentiment_better']].values

X = np.hstack([transcript_emb, audio_features, sentiment_features])

print("Target range:", np.min(y), "to", np.max(y))

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter search
params = {
    "n_estimators": [100, 200, 300],
    "max_depth": [4, 6, 8, 10],
    "min_samples_split": [2, 5, 10]
}

base_model = RandomForestRegressor(random_state=42, n_jobs=-1)
search = RandomizedSearchCV(base_model, params, n_iter=10, cv=3, scoring='neg_mean_squared_error')
search.fit(X_train, y_train)
print("Best params:", search.best_params_)

# Train model with best parameters
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

with open(model_path, 'wb') as f:
    pickle.dump(model, f)
print("Model saved to random_forest_views_rating_model.pkl")

# Feature importance plot
importances = model.feature_importances_
feature_names = (
    [f"emb_{i}" for i in range(transcript_emb.shape[1])]
    + list(df[['duration', 'rms', 'zcr', 'spectral_centroid', 'spectral_bandwidth', 'tempo']].columns)
    + ['transcript_sentiment', 'transcript_sentiment_better']
)
top_n = 10
indices = np.argsort(importances)[-top_n:]  # Top N features

plt.figure(figsize=(8, 5))
plt.barh(np.array(feature_names)[indices], importances[indices])
plt.xlabel("Importance")
plt.title(f"Top {top_n} Feature Importances")
plt.tight_layout()
plt.savefig(image_path)
plt.close()
print("Feature importance plot saved as feature_importance.png")
