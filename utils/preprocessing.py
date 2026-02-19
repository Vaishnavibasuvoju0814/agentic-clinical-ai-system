# utils/preprocessing.py

import pandas as pd
from sklearn.preprocessing import StandardScaler


def preprocess_data(df, target_column):

    # Remove duplicates
    df = df.drop_duplicates()

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Detect types
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = X.select_dtypes(include=["object"]).columns

    # Fill missing values
    X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
    X[categorical_cols] = X[categorical_cols].fillna("Unknown")

    # Encode categoricals
    X = pd.get_dummies(X, drop_first=True)

    # Scale numeric
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    return X, y
