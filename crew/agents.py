from crewai import Agent
from .llm_config import ollama_llm

# ---------------------------
# NON-LLM AGENTS
# ---------------------------

data_agent = Agent(
    role="Data Validation Specialist",
    goal="Validate and prepare clinical dataset.",
    backstory="Handles structured medical datasets.",
    llm=ollama_llm,  # 🔥 IMPORTANT
    allow_delegation=False,
    verbose=False,
)

feature_agent = Agent(
    role="Feature Engineering Specialist",
    goal="Perform feature engineering and remove correlations.",
    backstory="Expert in preprocessing medical features.",
    llm=ollama_llm,  # 🔥 IMPORTANT
    allow_delegation=False,
    verbose=False,
)

prediction_agent = Agent(
    role="Disease Prediction Specialist",
    goal="Train ML models and select best model.",
    backstory="Medical ML expert.",
    llm=ollama_llm,  # 🔥 IMPORTANT
    allow_delegation=False,
    verbose=False,
)

# ---------------------------
# LLM AGENTS (Ollama Mistral)
# ---------------------------

risk_agent = Agent(
    role="Risk Assessment Agent",
    goal="Interpret SHAP values and explain risk level.",
    backstory="Clinical AI explanation specialist.",
    llm=ollama_llm,
    allow_delegation=False,
    verbose=True,
)

report_agent = Agent(
    role="Medical Report Generation Agent",
    goal="Generate structured medical summary report.",
    backstory="Produces professional clinical documentation.",
    llm=ollama_llm,
    allow_delegation=False,
    verbose=True,
)