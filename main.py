from agents.data_agent import DataValidationAgent
from agents.feature_agent import FeatureSelectionAgent
from agents.prediction_agent import DiseasePredictionAgent
from agents.explainability_agent import ExplainabilityAgent
from agents.report_agent import ReportAgent


data_agent = DataValidationAgent()
feature_agent = FeatureSelectionAgent()
prediction_agent = DiseasePredictionAgent()
explain_agent = ExplainabilityAgent()
report_agent = ReportAgent()

datasets = [
    ("heart", "data/cardio.csv"),
    ("diabetes", "data/diabetes.csv"),
    ("ckd", "data/ckd.csv"),
]

print("\nMODEL TRAINING RESULTS")
print("=" * 60)

for disease, path in datasets:

    data_output = data_agent.run(disease, path)
    feature_output = feature_agent.run(data_output)
    prediction_output = prediction_agent.run(feature_output)
    explain_output = explain_agent.run(prediction_output)

    print(f"\nDisease: {disease}")
    print("Best Model:", prediction_output["best_model"])
    print("Accuracy:", round(
        max(prediction_output["lr_metrics"]["accuracy"],
            prediction_output["lgb_metrics"]["accuracy"]), 4))
    print("Risk Level:", explain_output["risk_level"])

    report_path = report_agent.run(explain_output, prediction_output)
    print("Report Generated:", report_path)
    print("-" * 60)
