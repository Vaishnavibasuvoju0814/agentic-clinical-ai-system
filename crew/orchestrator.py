from crewai import Crew, Task
from .agents import (
    data_agent,
    feature_agent,
    prediction_agent,
    risk_agent,
    report_agent
)

from agents.data_agent import DataValidationAgent
from agents.feature_agent import FeatureSelectionAgent
from agents.prediction_agent import DiseasePredictionAgent
from agents.explainability_agent import ExplainabilityAgent
from agents.report_agent import ReportAgent


class MedicalCrewOrchestrator:

    def __init__(self):
        self.data_tool = DataValidationAgent()
        self.feature_tool = FeatureSelectionAgent()
        self.prediction_tool = DiseasePredictionAgent()
        self.explain_tool = ExplainabilityAgent()
        self.report_tool = ReportAgent()

    def run(self, disease, dataset_path):

        # -----------------------
        # STEP 1: Data
        # -----------------------
        data_output = self.data_tool.run(disease, dataset_path)

        # -----------------------
        # STEP 2: Feature
        # -----------------------
        feature_output = self.feature_tool.run(data_output)

        # -----------------------
        # STEP 3: Prediction
        # -----------------------
        prediction_output = self.prediction_tool.run(feature_output)

        # -----------------------
        # STEP 4: SHAP (No LLM)
        # -----------------------
        explain_output = self.explain_tool.run(prediction_output)

        # -----------------------
        # STEP 5: LLM Risk Explanation
        # -----------------------
        risk_prompt = f"""
        Disease: {disease}
        Probability: {explain_output['probability']*100:.2f}%
        Risk Level: {explain_output['risk_level']}

        Explain briefly (3-4 lines) why this risk level was assigned 
        based on key contributing features.
        Keep it concise and clinical.
        """
        risk_task = Task(
            description=risk_prompt,
            agent=risk_agent,
            expected_output="Detailed clinical explanation."
        )

        # -----------------------
        # STEP 6: LLM Report Narrative
        # -----------------------
        report_prompt = f"""
        Generate a concise medical summary (3-4 lines) for:

        Disease: {disease}
        Best Model: {prediction_output['best_model']}
        Probability: {explain_output['probability']*100:.2f}%
        Risk Level: {explain_output['risk_level']}

        Provide professional clinical language.
        """

        report_task = Task(
            description=report_prompt,
            agent=report_agent,
            expected_output="Professional medical summary."
        )

        crew = Crew(
            agents=[risk_agent, report_agent],
            tasks=[risk_task, report_task],
            verbose=True
        )

        crew_output = crew.kickoff()

        # -----------------------
        # STEP 7: Generate HTML Report (Your Existing Tool)
        # -----------------------
        report_path = self.report_tool.run(
            explain_output,
            prediction_output
        )

        return {
            "prediction_output": prediction_output,
            "explain_output": explain_output,
            "llm_output": crew_output,
            "report_path": report_path
        }