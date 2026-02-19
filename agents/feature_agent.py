# agents/feature_agent.py

import numpy as np
import pandas as pd


class FeatureSelectionAgent:

    def run(self, input_data):

        X = input_data["X"]
        y = input_data["y"]
        disease = input_data["disease"]

        # ===== Additional Feature Engineering for Heart & Diabetes =====
        if disease in ["heart", "diabetes"]:

            # Add interaction terms
            numeric_cols = X.select_dtypes(include=["float64", "int64"]).columns

            for col in numeric_cols[:5]:  # limited to avoid explosion
                X[f"{col}_squared"] = X[col] ** 2

            if len(numeric_cols) >= 2:
                X["interaction_1"] = X[numeric_cols[0]] * X[numeric_cols[1]]

        # ===== Correlation Removal =====
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        to_drop = [
            column for column in upper.columns
            if any(upper[column] > 0.92)
        ]

        X_reduced = X.drop(columns=to_drop)

        return {
            "disease": disease,
            "X": X_reduced,
            "y": y
        }
