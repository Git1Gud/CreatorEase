import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

# Load cleaned data
df = pd.read_csv('youtube_shorts_podcast_dataset_cleaned.csv')

# Convert embedding columns from string to list of floats
import ast
df['question_embedding'] = df['question_embedding'].apply(ast.literal_eval)
df['answer_embedding'] = df['answer_embedding'].apply(ast.literal_eval)

# Combine embeddings (e.g., concatenate)
question_emb = np.vstack(df['question_embedding'].apply(np.array).to_list())
answer_emb = np.vstack(df['answer_embedding'].apply(np.array).to_list())
X_embed = np.hstack([question_emb, answer_emb])

# Add sentiment features
sentiment_features = df[['question_sentiment', 'answer_sentiment']].values
X = np.hstack([X_embed, sentiment_features])

# Target variable
target = 'views'  # Make sure your CSV has a 'views' column

# Drop rows with missing target
df = df.dropna(subset=[target])
y = np.log1p(df[target].values)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter tuning (RandomizedSearchCV)
params = {
    "n_estimators": [100, 200, 300],
    "max_depth": [4, 6, 8, 10],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.7, 0.8, 1.0]
}

model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
search = RandomizedSearchCV(model, params, n_iter=10, cv=3, scoring='neg_mean_squared_error')
search.fit(X_train, y_train)
print("Best params:", search.best_params_)

# Train model (XGBoost Regressor) with best parameters
model = xgb.XGBRegressor(**search.best_params_, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
y_pred = np.expm1(y_pred)
mse = mean_squared_error(np.expm1(y_test), y_pred)
r2 = r2_score(np.expm1(y_test), y_pred)
print(f"Test MSE: {mse:.2f}")
print(f"Test R2 Score: {r2:.2f}")