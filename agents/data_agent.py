# agents/data_agent.py

import pandas as pd
from utils.preprocessing import preprocess_data
from config.disease_config import DISEASE_CONFIG


class DataValidationAgent:

    def run(self, disease_name, dataset_path):

        # ======================================================
        # 1️⃣ Disease Validation
        # ======================================================
        if disease_name not in DISEASE_CONFIG:
            raise ValueError(f"Unsupported disease: {disease_name}")

        target_column = DISEASE_CONFIG[disease_name]["target"]

        # ======================================================
        # 2️⃣ Load Dataset
        # ======================================================
        df = pd.read_csv(dataset_path)

        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")

        # ======================================================
        # 3️⃣ Basic Data Validation
        # ======================================================
        total_rows = df.shape[0]
        total_columns = df.shape[1]

        missing_values = df.isnull().sum().sum()

        # Ensure binary classification
        unique_classes = df[target_column].unique()

        if len(unique_classes) != 2:
            raise ValueError(
                f"Target column must be binary. Found classes: {unique_classes}"
            )

        # ======================================================
        # 4️⃣ Preprocessing
        # ======================================================
        X, y = preprocess_data(df, target_column)

        class_distribution = y.value_counts().to_dict()

        # Class imbalance ratio
        majority = max(class_distribution.values())
        minority = min(class_distribution.values())
        imbalance_ratio = round(majority / minority, 2) if minority != 0 else 0

        # ======================================================
        # 5️⃣ Metadata for Downstream Agents
        # ======================================================
        metadata = {
            "rows": X.shape[0],
            "original_columns": total_columns,
            "features_after_processing": X.shape[1],
            "missing_values": int(missing_values),
            "class_distribution": class_distribution,
            "imbalance_ratio": imbalance_ratio
        }

        return {
            "disease": disease_name,
            "X": X,
            "y": y,
            "metadata": metadata
        }