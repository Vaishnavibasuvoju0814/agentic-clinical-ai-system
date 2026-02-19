# agents/explainability_agent.py

import shap
import numpy as np


class ExplainabilityAgent:

    def run(self, prediction_output):

        model = prediction_output["model"]
        X_test = prediction_output["X_test"]
        best_model = prediction_output["best_model"]
        disease = prediction_output["disease"]

        # Take first test patient
        sample = X_test.iloc[[0]]

        # ---------------- SHAP ----------------
        if best_model == "LightGBM":

            explainer = shap.TreeExplainer(model)
            shap_values = explainer(sample)

            # New SHAP format fix
            shap_vals = shap_values.values[0]

        else:

            explainer = shap.LinearExplainer(model, X_test)
            shap_vals = explainer.shap_values(sample)[0]

        feature_names = sample.columns

        shap_dict = {}
        contribution_dict = {}

        for i, feature in enumerate(feature_names):

            value = float(shap_vals[i])
            shap_dict[feature] = round(value, 4)

            if value > 0:
                contribution_dict[feature] = "High"
            else:
                contribution_dict[feature] = "Low"

        probability = model.predict_proba(sample)[0][1]

        if probability >= 0.7:
            risk = "High Risk"
        elif probability >= 0.4:
            risk = "Moderate Risk"
        else:
            risk = "Low Risk"

        return {
            "disease": disease,
            "probability": round(float(probability), 4),
            "risk_level": risk,
            "shap_values": shap_dict,
            "contributions": contribution_dict
        }
