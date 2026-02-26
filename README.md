
# 🏥 Agentic Clinical AI System

> Multi-Disease Clinical Risk Assessment System using ML Pipelines, SHAP Explainability, and CrewAI Orchestration.

---

## 🚀 Overview

**Agentic Clinical AI System** is a modular, production-style AI framework designed to:

* Perform disease risk prediction
* Apply feature engineering & preprocessing pipelines
* Select best ML model dynamically
* Generate SHAP-based explainability
* Orchestrate workflow using CrewAI agents
* Produce structured medical risk reports

This project demonstrates real-world ML system design with agent-based orchestration.

---

## 🧠 Supported Diseases

The system currently supports:

* ❤️ Cardiovascular Disease
* 🩸 Diabetes
* 🫁 (Add your third disease here if applicable)

Each disease has:

* Dedicated preprocessing pipeline
* Feature selection
* Model comparison
* Risk probability estimation
* SHAP interpretation

---

## 🏗️ System Architecture

```
Input Data
   ↓
Preprocessing Layer
   ↓
Feature Engineering
   ↓
Model Selection (LR / RF / LightGBM)
   ↓
Risk Prediction
   ↓
SHAP Explainability
   ↓
CrewAI Agent Orchestration
   ↓
Medical Report Generation
```

---

## 📂 Project Structure

```
agentic-clinical-ai-system/
│
├── config/              # Configuration settings
├── crew/                # CrewAI agent definitions
├── tools/               # ML tools & processing modules
├── data/                # Datasets
├── reports/             # Generated reports
│
├── main.py              # Entry point
├── requirements.txt     # Project dependencies
└── README.md
```

---

## ⚙️ Tech Stack

* Python 3.11
* Scikit-learn
* LightGBM
* SHAP
* Pandas / NumPy
* CrewAI
* Ollama (for LLM-based report interpretation)

---

## 🔬 Machine Learning Pipeline

Each disease pipeline includes:

* Missing value handling
* Encoding
* Feature scaling
* Feature selection (SelectKBest)
* Model comparison
* Best model selection
* Risk probability output
* SHAP interpretation

---

## 🤖 Agent Orchestration (CrewAI)

Agents are responsible for:

* Preprocessing management
* Model execution
* Risk analysis
* Report generation
* Explainability summarization

This makes the system modular and extensible.

---

## 📊 Model Comparison

Models evaluated:

* Logistic Regression
* Random Forest
* LightGBM

Best performing model selected dynamically per disease.

---

## 📝 Report Generation

System produces:

* Risk probability
* Risk level classification
* Top influencing features (SHAP)
* Clinical interpretation
* Structured medical summary

---

## 🛠️ Installation

Clone the repository:

```bash
git clone https://github.com/Vaishnavibasuvoju0814/agentic-clinical-ai-system.git
cd agentic-clinical-ai-system
```

Create virtual environment:

```bash
python -m venv venv
```

Activate:

**Windows**

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run The Project

```bash
python main.py
```

---

## 🎯 Key Features

✔ Modular ML pipelines
✔ Multi-disease support
✔ SHAP explainability
✔ Agent-based orchestration
✔ Clean production folder structure
✔ Extendable architecture

---

## 📌 Future Improvements

* Web interface (FastAPI / Streamlit)
* Docker deployment
* API endpoints
* Database integration
* Real-time clinical dashboard

---

## 👩‍💻 Author

**Vaishnavi Basuvoju**
AI/ML Developer

---

## ⭐ If You Like This Project

Give it a star on GitHub ⭐
It helps a lot!

---want.
