from crewai import LLM

ollama_llm = LLM(
    model="ollama/mistral",
    base_url="http://localhost:11434",
    provider="ollama"
)