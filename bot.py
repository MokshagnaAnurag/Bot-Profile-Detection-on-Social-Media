#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler

# ---------------------------
# Custom transformer to extract posting hour from timestamp
# ---------------------------
class PostingHourExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts the posting hour from a datetime column.
    If the timestamp column is missing or unparseable, returns a default value (0).
    """
    def __init__(self, time_column='Created At'):
        self.time_column = time_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        if self.time_column not in X_copy.columns:
            # If timestamp column is missing, return 0 for all rows
            return pd.DataFrame({'posting_hour': np.zeros(len(X_copy), dtype=int)})
        # Convert to datetime (errors become NaT)
        if not np.issubdtype(X_copy[self.time_column].dtype, np.datetime64):
            X_copy[self.time_column] = pd.to_datetime(X_copy[self.time_column], errors='coerce')
        # Extract the hour; if NaT then fill with 0
        X_copy['posting_hour'] = X_copy[self.time_column].dt.hour.fillna(0).astype(int)
        return X_copy[['posting_hour']]

# ---------------------------
# Custom transformer to select text column
# ---------------------------
class TextSelector(BaseEstimator, TransformerMixin):
    """
    Selects the text column from the DataFrame.
    """
    def __init__(self, key='Tweet'):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Fill missing text with an empty string
        return X[self.key].fillna("")

# ---------------------------
# Custom transformer to select numeric columns
# ---------------------------
class NumericSelector(BaseEstimator, TransformerMixin):
    """
    Selects numeric columns from the DataFrame.
    """
    def __init__(self, keys):
        self.keys = keys

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Fill missing numeric values with 0 and ensure they are floats
        return X[self.keys].fillna(0).astype(float)

# ---------------------------
# Main function: load data, prepare features, train, evaluate and export report to CSV
# ---------------------------
def main():
    # --- Configuration: update as needed ---
    DATASET_PATH = 'bot_detection_data.csv'  # <-- Replace with your CSV file path
    LABEL_COLUMN = 'Bot Label'         # Target variable (0 for human, 1 for bot)
    TEXT_COLUMN = 'Tweet'              # Column with the tweet text
    TIMESTAMP_COLUMN = 'Created At'    # Timestamp column
    NUMERIC_COLUMNS = ['Retweet Count', 'Mention Count', 'Follower Count', 'Verified']
    
    # Columns to include in the final CSV report
    REPORT_COLUMNS = ['User ID', 'Username', TEXT_COLUMN]

    # --- Load the dataset ---
    df = pd.read_csv(DATASET_PATH)

    # --- Verify required columns exist ---
    for col in [TEXT_COLUMN, LABEL_COLUMN]:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in the dataset.")

    # --- Convert label to numeric (assume binary: 0 for human, 1 for bot) ---
    df[LABEL_COLUMN] = pd.to_numeric(df[LABEL_COLUMN], errors='coerce')
    df = df.dropna(subset=[LABEL_COLUMN])

    # --- (Optional) Convert 'Verified' to numeric if needed ---
    if 'Verified' in NUMERIC_COLUMNS:
        df['Verified'] = pd.to_numeric(df['Verified'], errors='coerce').fillna(0)

    # --- Split the dataset into training and test sets ---
    X_train, X_test, y_train, y_test = train_test_split(
        df, df[LABEL_COLUMN],
        test_size=0.2,
        random_state=42,
        stratify=df[LABEL_COLUMN]
    )

    # --- Build the feature extraction pipelines ---
    # Text feature pipeline: using the Tweet column and TF-IDF
    text_pipeline = Pipeline([
        ('selector', TextSelector(key=TEXT_COLUMN)),
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2)))
    ])

    # Posting hour pipeline: extract hour from the Created At column
    time_pipeline = Pipeline([
        ('hour_extractor', PostingHourExtractor(time_column=TIMESTAMP_COLUMN))
    ])

    # Numeric features pipeline: select numeric columns and scale them
    numeric_pipeline = Pipeline([
        ('selector', NumericSelector(keys=NUMERIC_COLUMNS)),
        ('scaler', StandardScaler())
    ])

    # Combine all features using FeatureUnion
    combined_features = FeatureUnion(transformer_list=[
        ('text_features', text_pipeline),
        ('time_features', time_pipeline),
        ('numeric_features', numeric_pipeline)
    ])

    # --- Create the full pipeline ---
    pipeline = Pipeline([
        ('features', combined_features),
        ('clf', LogisticRegression(max_iter=1000, solver='liblinear'))
    ])

    # --- Train the model ---
    pipeline.fit(X_train, y_train)

    # --- Evaluate the model on the test set ---
    y_test_pred = pipeline.predict(X_test)
    y_test_prob = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline.named_steps['clf'], 'predict_proba') else None
    print("=== Classification Report on Test Set ===")
    print(classification_report(y_test, y_test_pred, digits=4))
    if y_test_prob is not None:
        roc_auc = roc_auc_score(y_test, y_test_prob)
        print(f"AUC-ROC: {roc_auc:.4f}\n")

    # --- Generate per-user report for the entire dataset ---
    predictions = pipeline.predict(df)
    probabilities = pipeline.predict_proba(df)[:, 1] if hasattr(pipeline.named_steps['clf'], 'predict_proba') else np.zeros(len(df))

    # Create a new DataFrame for the CSV report
    report_df = df[REPORT_COLUMNS].copy()
    report_df['Predicted Bot'] = predictions  # 0 = human, 1 = bot
    report_df['Bot Confidence'] = probabilities

    # Save the report DataFrame to a CSV file with proper formatting
    output_csv = 'bot_detection_report.csv'
    report_df.to_csv(output_csv, index=False)
    print(f"Per-user bot detection report saved to '{output_csv}'.")

if __name__ == '__main__':
    main()
