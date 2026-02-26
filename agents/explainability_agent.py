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

        # ======================================================
        # SHAP Explainer Selection
        # ======================================================

        if best_model in ["LightGBM", "Random Forest"]:

            explainer = shap.TreeExplainer(model)
            shap_values = explainer(sample)

            # Extract raw numpy values safely
            if hasattr(shap_values, "values"):
                shap_vals = shap_values.values
            else:
                shap_vals = shap_values

            # Convert to numpy array
            shap_vals = np.array(shap_vals)

            # If 3D (1, features, classes) → take class 1
            if shap_vals.ndim == 3:
                shap_vals = shap_vals[:, :, 1]

            # If list style → take class 1
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]

            # Now take first sample
            shap_vals = shap_vals[0]

        elif best_model == "Logistic Regression":

            explainer = shap.LinearExplainer(model, X_test)
            shap_vals = explainer.shap_values(sample)

            shap_vals = np.array(shap_vals)

            # If binary list → take class 1
            if shap_vals.ndim == 3:
                shap_vals = shap_vals[1]

            shap_vals = shap_vals[0]

        else:
            raise ValueError(f"Unsupported model type: {best_model}")

        # ======================================================
        # SHAP Processing
        # ======================================================

        feature_names = sample.columns

        shap_dict = {}
        contribution_dict = {}

        for i, feature in enumerate(feature_names):

            value = float(np.squeeze(shap_vals[i]))  # <-- FINAL SAFE FIX
            shap_dict[feature] = round(value, 4)

            if value > 0:
                contribution_dict[feature] = "High"
            else:
                contribution_dict[feature] = "Low"

        # ======================================================
        # Risk Calculation
        # ======================================================

        probability = model.predict_proba(sample)[0][1]

        if probability >= 0.7:
            risk = "High Risk"
        elif probability >= 0.4:
            risk = "Moderate Risk"
        else:
            risk = "Low Risk"

        return {
            "disease": disease,
            "best_model": best_model,
            "probability": round(float(probability), 4),
            "risk_level": risk,
            "shap_values": shap_dict,
            "contributions": contribution_dict
        }