# agents/feature_agent.py

import numpy as np
import pandas as pd


class FeatureSelectionAgent:

    def run(self, input_data):

        X = input_data["X"].copy()   # ✅ Avoid modifying original
        y = input_data["y"]
        disease = input_data["disease"]

        original_feature_count = X.shape[1]

        # ======================================================
        # 1️⃣ Controlled Feature Engineering
        # ======================================================
        numeric_cols = X.select_dtypes(include=["float64", "int64"]).columns.tolist()

        # Add squared terms (limit to first 3 numeric columns)
        for col in numeric_cols[:3]:
            X[f"{col}_squared"] = X[col] ** 2

        # Add one interaction term only (avoid explosion)
        if len(numeric_cols) >= 2:
            X["interaction_1"] = X[numeric_cols[0]] * X[numeric_cols[1]]

        # ======================================================
        # 2️⃣ Correlation Removal (Important for LR stability)
        # ======================================================
        corr_matrix = X.corr().abs()

        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        to_drop = [
            column for column in upper.columns
            if any(upper[column] > 0.90)   # slightly stricter
        ]

        X_reduced = X.drop(columns=to_drop)

        # ======================================================
        # 3️⃣ Feature Metadata
        # ======================================================
        metadata = {
            "original_features": original_feature_count,
            "engineered_features_added": X.shape[1] - original_feature_count,
            "features_removed_due_to_correlation": len(to_drop),
            "final_feature_count": X_reduced.shape[1]
        }

        return {
            "disease": disease,
            "X": X_reduced,
            "y": y,
            "feature_metadata": metadata
        }