# ML-Powered Research Assistant

The **ML-Powered Research Assistant** is a web application designed to assist researchers by analyzing indiviual and multiple PDF research documents. It provides individual summaries, a comparative final summary, sentiment analysis, keyword extraction, and a RAG-powered chatbot to answer questions about the documents. The backend is built with FastAPI and leverages machine learning models for natural language processing, while the frontend is developed using Next.js with TypeScript and Tailwind CSS for a modern, responsive UI.

## Features
- **Upload Multiple PDFs**: Upload multiple research documents (PDFs) for analysis.
- **Individual Summaries**: Generate concise summaries for each uploaded document.
- **Final Summary**: Compare key themes and findings across all documents in a comparative summary.
- **Sentiment Analysis**: Analyze the sentiment of the summaries using a pre-trained DistilBERT model.
- **Keyword Extraction**: Extract relevant keywords from each document using spaCy.
- **RAG-Powered Chatbot**: Chat with a Retrieval-Augmented Generation (RAG) chatbot to ask questions about the documents, powered by LLMs and embeddings.

## Tech Stack
### Backend
- **Framework**: FastAPI (Python)
- **LLM**: Ollama (LLaMA 3.3 for text generation, Nomic Embed for embeddings)
- **NLP Libraries**:
  - `llama-index`: For document indexing, summarization, and querying.
  - `spaCy`: For keyword extraction.
  - `transformers`: For sentiment analysis using DistilBERT.
- **PDF Processing**: PyPDF2
- **Environment Management**: `python-dotenv` for environment variables

### Frontend
- **Framework**: Next.js (React with TypeScript)
- **Styling**: Tailwind CSS
- **Components**: Custom React components (`chat-interface.tsx`, `document-uploader.tsx`, `summary-panel.tsx`, `theme-provider.tsx`)
- **State Management**: React hooks
- **Build Tools**: TypeScript, PostCSS, ESLint

## Prerequisites
- **Python**: 3.8+ (for the backend)
- **Node.js**: 18+ (for the frontend)
- **Git**: For cloning the repository
- **Ollama Server**: Access to an Ollama server for LLM and embedding models (or a local setup)

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/abm1499/ML-Powered-Research-Assistant.git
cd ML-Powered-Research-Assistant
```
### 2. Backend Setup
Navigate to the backend directory:
```bash
cd backend
```

Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

If requirements.txt is missing, install the required packages manually:
```bash
pip install fastapi uvicorn llama-index langchain-community pypdf2 requests python-dotenv spacy transformers torch
python -m spacy download en_core_web_sm
```

Create a .env file in the backend directory with the following:
```bash
LLM_API_URL=<your-ollama-llm-api-url>
EMBEDDING_API_URL=<your-ollama-embedding-api-url>
```

Replace <your-ollama-llm-api-url> and <your-ollama-embedding-api-url> with the URLs of your Ollama server (e.g., http://localhost:11434 if running locally).

Run the backend server:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Frontend Setup
Navigate to the frontend directory:
```bash
cd frontend
```

Install dependencies:
```bash
npm install
```

Run the frontend development server:
```bash
npm run dev
```

## Usage

#### 1. Open the frontend in your browser (http://localhost:3000).
#### 2. Use the document uploader to upload one or more PDF research documents.
#### 3. View the analysis results:
Summaries: Individual summaries for each document.

Final Summary: A comparative summary highlighting key themes and differences.

Sentiment Analysis: Sentiment of the summaries (positive, negative, or neutral).

Keywords: Extracted keywords from each document.
#### 4. Use the chat interface to ask questions about the documents (e.g., "What are the main findings?").

