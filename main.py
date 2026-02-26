import sys
from crew.orchestrator import MedicalCrewOrchestrator


datasets = {
    "heart": "data/cardio.csv",
    "diabetes": "data/diabetes.csv",
    "ckd": "data/ckd.csv",
}


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python main.py [heart|diabetes|ckd]")
        sys.exit(1)

    disease = sys.argv[1]

    if disease not in datasets:
        print("Invalid disease selection.")
        sys.exit(1)

    orchestrator = MedicalCrewOrchestrator()

    result = orchestrator.run(disease, datasets[disease])

    prediction = result["prediction_output"]
    explain = result["explain_output"]

    print("\n=======================================")
    print("Best Model:", prediction["best_model"])
    print("Probability:", explain["probability"]*100, "%")
    print("Risk Level:", explain["risk_level"])
    print("Report Generated:", result["report_path"])
    print("=======================================\n")