🏥 Agentic Clinical AI Framework

A hybrid multi-disease clinical decision support system combining traditional machine learning, SHAP-based explainability, and CrewAI multi-agent orchestration with local LLM reasoning via Ollama (Mistral).

📌 Overview

This project implements a modular clinical AI system designed to:

Train and compare multiple machine learning models

Automatically select the best model based on AUC

Generate SHAP-based interpretability outputs

Use a local LLM (Mistral via Ollama) for clinical reasoning

Produce structured medical summary reports

The system demonstrates a production-style hybrid architecture integrating deterministic ML pipelines with LLM-based explanation.

🧠 System Architecture

The system is organized into independent agents orchestrated by CrewAI.

1️⃣ Data Validation Agent

Validates dataset integrity

Ensures correct target configuration

Prepares structured inputs

2️⃣ Feature Engineering Agent

Applies feature transformations

Removes highly correlated features

Prepares optimized model input

3️⃣ Prediction Agent

Trains and compares:

Logistic Regression

Random Forest

LightGBM

Selects the best model using AUC as the primary metric.

4️⃣ Risk Assessment Agent (LLM – Mistral)

Interprets SHAP contributions

Explains assigned risk level

Generates concise clinical reasoning

5️⃣ Report Generation Agent (LLM – Mistral)

Produces structured medical summaries

Generates clinician-friendly interpretation

Creates HTML report

🏥 Supported Diseases

Heart Disease

Diabetes

Chronic Kidney Disease (CKD)

⚙️ Technology Stack

Python 3.11.x

CrewAI

Ollama (Mistral)

scikit-learn

LightGBM

SHAP

Pandas

NumPy

Jinja2

Matplotlib

Seaborn

📂 Project Structure
AgenticAI-framework/
│
├── agents/                    # Core ML & explainability logic
│   ├── data_agent.py
│   ├── feature_agent.py
│   ├── prediction_agent.py
│   ├── explainability_agent.py
│   └── report_agent.py
│
├── crew/                      # CrewAI orchestration layer
│   ├── agents.py
│   ├── orchestrator.py
│   └── llm_config.py
│
├── config/                    # Disease configuration files
│
├── utils/                     # Preprocessing & helper utilities
│
├── data/                      # Input datasets
│
├── reports/                   # Generated HTML reports
│
├── main.py                    # Entry point
└── requirements.txt
🚀 Installation
Step 1 — Clone Repository
git clone https://github.com/yourusername/agentic-clinical-ai-system.git
cd agentic-clinical-ai-system
Step 2 — Create Virtual Environment
python -m venv venv
venv\Scripts\activate   # Windows
Step 3 — Install Dependencies
pip install -r requirements.txt
Step 4 — Install Ollama & Pull Mistral

Install Ollama from:

https://ollama.com

Then run:

ollama pull mistral

Ensure the Ollama server is running.

▶️ Running the System

Execute for any supported disease:

python main.py heart
python main.py diabetes
python main.py ckd
📊 Output

Each execution produces:

Model performance comparison

Best model selection (AUC-based)

Predicted probability

Risk classification

SHAP explanation

LLM-generated clinical interpretation

HTML medical report saved in /reports

🎯 Key Capabilities

✔ Multi-disease support
✔ Automatic model comparison & selection
✔ SHAP-based explainability
✔ Local LLM reasoning (no external API)
✔ Modular CrewAI orchestration
✔ Structured medical reporting

🔮 Future Improvements

Human-in-the-loop validation

Model calibration optimization

Web interface integration

Continuous learning module
