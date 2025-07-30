# setup.ps1 - for HP_RAG_PIPELINE
# Prerequisite: Python 3.13 must be installed and accessible via py -3.13

Write-Host "🛠️ Creating virtual environment with Python 3.13..."
py -3.13 -m venv rag_env

Write-Host "✅ Virtual environment created in 'rag_env'"

Write-Host "🔁 Activating virtual environment..."
.\rag_env\Scripts\activate

Write-Host "⬆️ Upgrading pip..."
pip install --upgrade pip

Write-Host "📦 Installing main dependencies..."
pip install `
    langchain `
    langchain-core `
    langchain-openai `
    langchain-groq `
    faiss-cpu `
    pypdf `
    langchain-community `
    python-dotenv `
    datasets `
    ragas `
    ipython `
    xxhash>=3.5.0 `
    "langgraph-checkpoint>=2.1.0,<3.0.0" `
    "langgraph-prebuilt>=0.6.0,<0.7.0" `
    "langgraph-sdk>=0.2.0,<0.3.0"

Write-Host "🔗 Installing LangGraph from GitHub..."
pip install git+https://github.com/langchain-ai/langgraph.git#subdirectory=libs/langgraph

Write-Host "📌 Freezing final dependencies to requirements.txt..."
pip freeze > requirements.txt

Write-Host "✅ Setup complete! Activate your environment with:"
Write-Host "`t .\rag_env\Scripts\activate"
