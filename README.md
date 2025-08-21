A fully functional Retrieval-Augmented Generation (RAG) system that integrates LangChain, LangGraph, FAISS, and RAGAS evaluation with Hugging Face embeddings and GPT-4o (via OpenRouter). 
This project demonstrates how to build a production-ready pipeline for document ingestion, semantic search, retrieval, and response generation, along with evaluation metrics to measure accuracy and reliability.

---

## 🏗️ Architecture Overview

```mermaid
flowchart LR
    %% Document Ingestion
    A[ Upload Docs] --> A1[ Text Splitter\nChunking into passages]
    A1 --> B[ Hugging Face Embeddings -sentence-transformers/all-MiniLM-L6-v2]
    
    %% Vector Database
    B --> C[🗄 FAISS Vector Store]
    C --> C1[( Similarity Search\nTop-k retrieval)]
    
    %% Retriever
    C1 --> D[⚙ Retriever - LangChain]
    D --> D1[ Relevant Chunks]
    
    %% Workflow
    D1 --> E[ LangGraph Workflow]
    E --> E1[ Routing\n(choose tools, handle branches)]
    E1 --> E2[ Combine context with query]
    
    %% LLM Interaction
    E2 --> F[ GPT-4o via OpenRouter]
    F --> F1[ Generate Contextual Answer]
    
    %% Output
    F1 --> G[ Final Answer to User]
    
    %% Evaluation
    G --> H[RAGAS Evaluation]
    H --> H1[ Faithfulness Check]
    H --> H2[ Answer Relevance]
    H --> H3[ Context Recall]

```
--------------------
## 🔑 Key Components

* **LangChain** – Builds the RAG pipeline (retrieval + generation).
* **LangGraph** – Adds structured workflows for better orchestration.
* **FAISS** – Vector database for efficient similarity search.
* **Hugging Face Embeddings** – Converts docs into vectors.
* **OpenRouter (GPT-4o)** – Free access to GPT-4o for LLM answers.
* **RAGAS** – Evaluates retrieval accuracy & faithfulness.

---

## ⚡ Quickstart

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Mshroom/RAG-system-for-Harry-Potter-Book.git
cd complex-RAG-guide
```

### 2️⃣ Create Virtual Environment & Install Dependencies

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

pip install -r requirements.txt
```

### 3️⃣ Add API Keys

Create a `.env` file in the root directory:

```
OPENROUTER_API_KEY=your_api_key_here
```

👉 Get your free API key from [OpenRouter](https://openrouter.ai).

---

## ▶️ Beginner Tutorial

Here’s a minimal example to **embed documents, store them in FAISS, and query GPT-4o**.

```python
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
import os

# Load API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENROUTER_API_KEY")

# Step 1: Define documents
docs = [
    "Harry Potter is a wizard who studied at Hogwarts.",
    "Hogwarts is a school of magic in Scotland."
]

# Step 2: Create embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Step 3: Store embeddings in FAISS
db = FAISS.from_texts(docs, embedding_model)

# Step 4: Initialize GPT-4o via OpenRouter
llm = ChatOpenAI(model="openai/gpt-4o", openai_api_base="https://openrouter.ai/api/v1")

# Step 5: Build Retrieval-QA chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever()
)

# Step 6: Ask a question
query = "Who is Harry Potter?"
answer = qa.run(query)

print("Q:", query)
print("A:", answer)


## 📊 Evaluating with RAGAS

Once you’ve tested your pipeline, evaluate it with RAGAS:

```python
from ragas.metrics import faithfulness, answer_relevance, context_recall
from ragas import evaluate

# Example QA pairs for evaluation
examples = [
    {"question": "Who is Harry Potter?",
     "answer": "Harry Potter is a wizard who studied at Hogwarts.",
     "contexts": ["Harry Potter is a wizard who studied at Hogwarts."]}
]

# Run evaluation
result = evaluate(examples, metrics=[faithfulness, answer_relevance, context_recall])
print(result)
```

---

## 📂 Project Structure

```
complex-RAG-guide/
│── main.py              # Main RAG pipeline
│── helper_functions.py       # RAGAS evaluation script
│── requirements.txt    # Dependencies
│── .env.example        # Example API key setup
│── data/               # Folder for input documents
│── vectorstore/        # Stored FAISS index
```



## 📚 Resources

* [LangChain Docs](https://python.langchain.com)
* [LangGraph](https://www.langchain.com/langgraph)
* [FAISS](https://faiss.ai)
* [RAGAS](https://docs.ragas.io)
* [OpenRouter](https://openrouter.ai)

---

⚡ With this repo, you’ll build, query, and evaluate a complete **end-to-end RAG system** in just a few steps.
