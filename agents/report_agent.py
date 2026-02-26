# agents/report_agent.py

import os
from datetime import datetime


class ReportAgent:

    def run(self, explain_output, prediction_output):

        disease = explain_output["disease"].capitalize()
        probability = explain_output["probability"] * 100
        risk = explain_output["risk_level"]
        shap_values = explain_output["shap_values"]
        contributions = explain_output["contributions"]

        # Metrics
        lr_metrics = prediction_output["lr_metrics"]
        rf_metrics = prediction_output["rf_metrics"]
        lgb_metrics = prediction_output["lgb_metrics"]

        best_model = prediction_output["best_model"]
        best_metrics = prediction_output["best_metrics"]

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # ======================================================
        # SHAP Table (Top 10 Features)
        # ======================================================
        rows = ""
        for feature in list(shap_values.keys())[:10]:
            rows += f"""
            <tr>
                <td>{feature}</td>
                <td>{shap_values[feature]}</td>
                <td>{contributions[feature]}</td>
            </tr>
            """

        # ======================================================
        # Clinical Interpretation
        # ======================================================
        interpretation = f"""
        The patient shows a <b>{risk}</b> for {disease} with a predicted probability 
        of <b>{probability:.2f}%</b>. The optimal predictive model selected was 
        <b>{best_model}</b> based on AUC performance comparison. 
        SHAP-based explainability highlights the most influential physiological 
        parameters contributing to this risk estimation.

        This AI-generated assessment is intended to support clinical 
        decision-making and should be reviewed by qualified healthcare professionals.
        """

        # ======================================================
        # HTML Report
        # ======================================================
        html = f"""
        <html>
        <head>
            <title>{disease} Medical Assessment Report</title>
            <style>
                body {{
                    font-family: Arial;
                    margin: 40px;
                    background-color: #f8f9fa;
                }}
                h1 {{
                    color: #2c3e50;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-top: 20px;
                }}
                th, td {{
                    border: 1px solid #ccc;
                    padding: 8px;
                    text-align: center;
                }}
                th {{
                    background-color: #2980b9;
                    color: white;
                }}
                .highlight {{
                    background-color: #d4edda;
                    font-weight: bold;
                }}
                .prob-box {{
                    margin-top: 30px;
                    padding: 20px;
                    border: 2px solid #e74c3c;
                    font-size: 20px;
                    text-align: center;
                    background-color: #fff;
                }}
                .section {{
                    margin-top: 30px;
                }}
            </style>
        </head>
        <body>

        <h1>Medical Assessment Report</h1>
        <p><i>Generated on: {timestamp}</i></p>

        <div class="section">
            <h2>Disease: {disease}</h2>
            <p><b>Best Model Selected:</b> {best_model}</p>
        </div>

        <div class="section">
            <h2>Model Performance Comparison</h2>
            <table>
                <tr>
                    <th>Model</th>
                    <th>Accuracy</th>
                    <th>Recall</th>
                    <th>F1 Score</th>
                    <th>AUC</th>
                </tr>

                <tr class="{'highlight' if best_model == 'Logistic Regression' else ''}">
                    <td>Logistic Regression</td>
                    <td>{lr_metrics['accuracy']:.4f}</td>
                    <td>{lr_metrics['recall']:.4f}</td>
                    <td>{lr_metrics['f1']:.4f}</td>
                    <td>{lr_metrics['auc']:.4f}</td>
                </tr>

                <tr class="{'highlight' if best_model == 'Random Forest' else ''}">
                    <td>Random Forest</td>
                    <td>{rf_metrics['accuracy']:.4f}</td>
                    <td>{rf_metrics['recall']:.4f}</td>
                    <td>{rf_metrics['f1']:.4f}</td>
                    <td>{rf_metrics['auc']:.4f}</td>
                </tr>

                <tr class="{'highlight' if best_model == 'LightGBM' else ''}">
                    <td>LightGBM</td>
                    <td>{lgb_metrics['accuracy']:.4f}</td>
                    <td>{lgb_metrics['recall']:.4f}</td>
                    <td>{lgb_metrics['f1']:.4f}</td>
                    <td>{lgb_metrics['auc']:.4f}</td>
                </tr>

            </table>
        </div>

        <div class="section">
            <h2>Physiological Data & SHAP Contributions</h2>
            <table>
                <tr>
                    <th>Feature</th>
                    <th>SHAP Value</th>
                    <th>Contribution</th>
                </tr>
                {rows}
            </table>
        </div>

        <div class="prob-box">
            Predicted Probability: {probability:.2f}% <br>
            Risk Level: {risk}
        </div>

        <div class="section">
            <h2>Interpretation</h2>
            <p>{interpretation}</p>
        </div>

        </body>
        </html>
        """

        os.makedirs("reports", exist_ok=True)
        file_path = f"reports/{disease.lower()}_report.html"

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(html)

        return file_path