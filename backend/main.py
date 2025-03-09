import os
import PyPDF2
import requests
import asyncio
import hashlib
from dotenv import load_dotenv
from typing import List, Dict, Optional, Any
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from llama_index.core import SimpleDirectoryReader, Document, SummaryIndex, Settings, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.llms import CustomLLM, LLMMetadata, CompletionResponse
from langchain_community.embeddings import OllamaEmbeddings
from llama_index.core.llms.callbacks import llm_completion_callback
import spacy
from transformers import pipeline, AutoTokenizer

app = FastAPI()

load_dotenv()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("FRONTEND_URL", "http://localhost:3001")],  # frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# models
class OllamaLLaMA(CustomLLM):
    model_name: str
    api_url: str

    def __init__(self, model_name: str, api_url: str):
        super().__init__(model_name=model_name, api_url=api_url)

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=4096,
            num_output=512,
            model_name=self.model_name,
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        data = {"model": self.model_name, "prompt": prompt, "stream": False, "options": {"temperature": 0.7}}
        response = requests.post(f"{self.api_url}/api/generate", json=data)
        try:
            text = response.json().get("response", "")
            return CompletionResponse(text=text)
        except requests.exceptions.JSONDecodeError:
            return CompletionResponse(text=response.text)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs) -> Any:
        data = {"model": self.model_name, "prompt": prompt, "stream": False, "options": {"temperature": 0.7}}
        response = requests.post(f"{self.api_url}/api/generate", json=data)
        try:
            text = response.json().get("response", "")
            yield CompletionResponse(text=text)
        except requests.exceptions.JSONDecodeError:
            yield CompletionResponse(text=response.text)

llm = OllamaLLaMA(model_name="llama3.3:latest", api_url=os.getenv("LLM_API_URL"))
embed_model = OllamaEmbeddings(model="nomic-embed-text:latest", base_url=os.getenv("EMBEDDING_API_URL"))

Settings.llm = llm
Settings.embed_model = embed_model

nlp = spacy.load("en_core_web_sm")

sentiment_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    truncation=True,
    max_length=512,
    device=-1
)

# State management
state: Dict[str, any] = {
    "documents": [],
    "summaries": [],
    "final_summary": None,
    "keywords": [],
    "index": None,
    "query_engine": None
}

def hash_text(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()

def extract_text_from_pdf(pdf_path: str) -> str:
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = "".join(page.extract_text() + "\n" for page in pdf_reader.pages)
        return text.strip()
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return ""

async def generate_all_summaries(documents: List[Document]) -> List[str]:
    summary_index = SummaryIndex(documents)
    query_engine = summary_index.as_query_engine(response_mode="tree_summarize", use_async=True)
    summaries = []
    for doc in documents:
        response = await query_engine.aquery(f"Summarize the document with ID {doc.metadata['doc_id']} in a well structured way")
        summary = "\n".join(f"- {line.strip()}" for line in response.response.split("\n") if line.strip())
        summaries.append(summary)
    return summaries

async def generate_final_summary(summaries: List[str]) -> str:
    combined_text = "\n".join(summaries)
    doc = Document(text=combined_text)
    summary_index = SummaryIndex([doc])
    query_engine = summary_index.as_query_engine(response_mode="tree_summarize", use_async=True)
    response = await query_engine.aquery("Summarize and compare the key themes and findings of these documents in a well structured way")
    final_summary = "\n".join(f"- {line.strip()}" for line in response.response.split("\n") if line.strip())
    return final_summary

def analyze_sentiment(text: str) -> Dict[str, any]:
    encoded = sentiment_tokenizer.encode(text, add_special_tokens=True, max_length=512, truncation=True)
    truncated_text = sentiment_tokenizer.decode(encoded, skip_special_tokens=True)
    return sentiment_analyzer(truncated_text)[0]

def extract_keywords(text: str) -> List[str]:
    doc = nlp(text)
    keywords = [token.text for token in doc if token.is_alpha and not token.is_stop and 
                (token.pos_ in ["NOUN", "ADJ"] or token.ent_type_)]
    from collections import Counter
    keyword_counts = Counter(keywords)
    return [word for word, _ in keyword_counts.most_common(10)]

# API Endpoints
@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    state["documents"] = []
    for file in files:
        file_path = f"temp_{file.filename}"
        with open(file_path, "wb") as f:
            f.write(await file.read())
        text = extract_text_from_pdf(file_path)
        doc = Document(text=text, metadata={"doc_id": file.filename})
        state["documents"].append(doc)
        os.remove(file_path)

    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    nodes = splitter.get_nodes_from_documents(state["documents"])
    state["index"] = VectorStoreIndex(nodes, embed_model=embed_model)
    state["query_engine"] = state["index"].as_query_engine(similarity_top_k=4, response_mode="compact")
    
    return {"message": "Files uploaded and indexed successfully"}

@app.get("/summaries")
async def get_summaries():
    if not state["documents"]:
        raise HTTPException(status_code=400, detail="No documents uploaded")
    state["summaries"] = await generate_all_summaries(state["documents"])
    return {"summaries": state["summaries"]}

@app.get("/final-summary")
async def get_final_summary():
    if not state["summaries"]:
        raise HTTPException(status_code=400, detail="Summaries not generated")
    state["final_summary"] = await generate_final_summary(state["summaries"])
    return {"final_summary": state["final_summary"]}

@app.get("/sentiment")
async def get_sentiment():
    if not state["summaries"]:
        raise HTTPException(status_code=400, detail="Summaries not generated")
    text_to_analyze = state["final_summary"] if len(state["summaries"]) > 1 else state["summaries"][0]
    sentiment = analyze_sentiment(text_to_analyze)
    return {"sentiment": sentiment}

@app.get("/keywords")
async def get_keywords():
    if not state["documents"]:
        raise HTTPException(status_code=400, detail="No documents uploaded")
    state["keywords"] = [extract_keywords(doc.text) for doc in state["documents"]]
    return {"keywords": state["keywords"]}

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
async def query_documents(request: QueryRequest):
    if not state["query_engine"]:
        raise HTTPException(status_code=400, detail="No documents indexed")
    response = state["query_engine"].query(request.query)
    formatted_response = "\n".join(f"- {line.strip()}" for line in response.response.split("\n") if line.strip())
    return {"answer": formatted_response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)