# agents/data_agent.py

import pandas as pd
from utils.preprocessing import preprocess_data
from config.disease_config import DISEASE_CONFIG


class DataValidationAgent:

    def run(self, disease_name, dataset_path):

        if disease_name not in DISEASE_CONFIG:
            raise ValueError(f"Unsupported disease: {disease_name}")

        target_column = DISEASE_CONFIG[disease_name]["target"]

        df = pd.read_csv(dataset_path)

        X, y = preprocess_data(df, target_column)

        return {
            "disease": disease_name,
            "X": X,
            "y": y,
            "metadata": {
                "rows": X.shape[0],
                "features": X.shape[1],
                "class_distribution": y.value_counts().to_dict()
            }
        }
