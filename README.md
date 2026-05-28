#  chat-with-medical-pdf

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/LangChain-🦜-121212?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Mistral_AI-F7631B?style=for-the-badge&logo=mistral&logoColor=white"/>
  <img src="https://img.shields.io/badge/ChromaDB-8A2BE2?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white"/>
  <img src="https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge"/>
</p>
<p align="center">
  <strong>An AI-powered medical document assistant that lets you chat with any medical PDF using Retrieval-Augmented Generation.</strong><br/>
  Upload a document, ask questions, and get context-grounded answers — no hallucinations, just evidence-based responses.
</p>

         ⚠️ Disclaimer: This tool is strictly for educational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a licensed medical professional for clinical decisions.

<details>
<summary>📋 Table of Contents</summary>

- [About the Project](#-about-the-project)
- [How It Works](#-how-it-works)
- [Features](#-features)
- [RAG Pipeline Architecture](#-rag-pipeline-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Environment Variables](#environment-variables)
- [Usage](#-usage)
- [Screenshots](#-screenshots)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License](#-license)

</details>

**About the Project**
chat-with-medical-pdf is a Retrieval-Augmented Generation (RAG) system built for medical students and healthcare learners. Instead of relying on a plain LLM that might hallucinate medical facts, this system retrieves the most relevant passages directly from a medical document and uses them as grounded context for every answer.
The default knowledge base is the Gale Encyclopedia of Medicine (2nd Edition) — a comprehensive medical reference covering thousands of conditions, procedures, and terminology. Users can also upload their own medical PDFs at any time and chat with them instantly.
The goal is simple: make medical knowledge searchable, conversational, and accessible — while always staying grounded in source documents.

**How It Works**

A medical PDF is loaded and split into overlapping text chunks.
Each chunk is converted into a semantic vector using OpenAI Embeddings.
Vectors are stored and indexed in a ChromaDB vector database.
When a user asks a question, the query is embedded and matched against stored vectors.
The top relevant chunks are retrieved using MMR (Maximum Marginal Relevance) search.
Retrieved context + chat history + the question are passed to Mistral AI.
Mistral generates a grounded, medically accurate response.
The conversation history is maintained across turns for multi-turn dialogue.

**Features**

📄 Upload any medical PDF and chat with it instantly in the same session.
🏥 Default knowledge base from the Gale Encyclopedia of Medicine.
🧠 Conversational memory —> remembers context across multiple turns.
🔍 MMR-based retrieval —> fetches diverse, relevant document chunks.
🚫 Strict guardrails —> never diagnoses, always recommends professionals.
🔄 Seamless source switching —> toggle between uploaded PDF and default DB.
🎨 Clean minimal UI —> typographic black-and-white Streamlit interface.
⚡ Fast in-memory indexing —> uploaded PDFs are processed and ready instantly.

**RAG Pipleine Architecture**
![RAG Pipeline](screenshots/rag-pipeline.png)

**Tech Stack**

| Layer | Technology |
|-------|------------|
| **LLM** | Mistral AI (`mistral-small-2506`) |
| **Embeddings** | OpenAI Embeddings (`text-embedding-ada-002`) |
| **Vector Store** | ChromaDB |
| **RAG Framework** | LangChain |
| **Frontend** | Streamlit |
| **PDF Parsing** | PyPDF |
| **Environment** | Python-dotenv |

# Project Structure
chat-with-medical-pdf/
│
├── app.py                  # Streamlit UI — chat interface, file upload, session state
├── main.py                 # Core RAG logic — retriever, LLM, prompt, chat loop
├── database.py             # PDF ingestion, chunking, embedding, ChromaDB creation
│
├── Chroma_db/              # Persisted ChromaDB vector store (auto-generated)
│
├── .env                    # API keys 
├── .gitignore              # Excludes .env, Chroma_db, __pycache__, etc.
├── requirements.txt        # All Python dependencies
└── README.md               

**🚀 Getting Started**
Prerequisites:
-Python 3.10 or higher
-An OpenAI API key (for embeddings)
-A Mistral AI API key (for the LLM)

# Installation
1. Clone the repository
   git clone https://github.com/your-username/chat-with-medical-pdf.git
   cd chat-with-medical-pdf
2.Create and activate a virtual environment
   python -m venv venv
   # On macOS/Linux
   source venv/bin/activate
   # On Windows
   venv\Scripts\activate
3. Install dependencies
   pip install -r requirements.txt
4. Build the default vector database
   Place your medical PDF (e.g. The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf) in the project root, then run:
    python database.py
   This will chunk the PDF and persist the ChromaDB vector store to Chroma_db/.
   
**Environment Variables**
Create a .env file in the project root:
OPENAI_API_KEY=your_openai_api_key_here
MISTRAL_API_KEY=your_mistral_api_key_here

# Usage
Run the Streamlit app:
--streamlit run app.py
Then open your browser at http://localhost:8501
Or
run the terminal version
--python main.py
Type your question and press Enter. Type 0 to exit.

**Screenshots**

<img width="1372" height="900" alt="image" src="https://github.com/user-attachments/assets/f7f96590-e0d2-48aa-ae9c-720dd220037b" />
<img width="1176" height="138" alt="image" src="https://github.com/user-attachments/assets/dbcf021c-b80c-4426-8261-42357cd04d22" />

**License**
This project is licensed under the MIT License — see the LICENSE file for details.

**⚠️ Medical Disclaimer**
This application is designed solely for educational and research purposes. It does not provide medical advice. The information provided by this assistant should never be used as a substitute for professional medical judgment, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.

