import spacy
import os
import PyPDF2
import requests
import nest_asyncio
import asyncio
import streamlit as st
from typing import Optional, Any, List
from llama_index.core import SimpleDirectoryReader, Document, SummaryIndex, Settings, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.core.llms import CustomLLM, LLMMetadata, CompletionResponse
from transformers import pipeline, AutoTokenizer
from langchain_community.embeddings import OllamaEmbeddings
import hashlib

nest_asyncio.apply()

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
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        data = {"model": self.model_name, "prompt": prompt, "stream": False, "options": {"temperature": 0.7}}
        response = requests.post(f"{self.api_url}/api/generate", json=data)
        try:
            text = response.json().get("response", "")
            return CompletionResponse(text=text)
        except requests.exceptions.JSONDecodeError:
            return CompletionResponse(text=response.text)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> Any:
        raise NotImplementedError("Streaming not implemented yet.")

# Global models
llm = OllamaLLaMA(model_name="llama3.3:latest", api_url="http://fzi-gpu-03:11434")
embed_model = OllamaEmbeddings(model="nomic-embed-text:latest", base_url="http://fzi-gpu-03:11434")
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
        doc_hash = hash_text(doc.text)
        if f"summary_{doc_hash}" in st.session_state:
            summaries.append(st.session_state[f"summary_{doc_hash}"])
        else:
            response = await query_engine.aquery(f"Summarize the document with ID {doc.metadata['doc_id']}")
            summary = response.response
            st.session_state[f"summary_{doc_hash}"] = summary
            summaries.append(summary)
    return summaries

async def generate_final_summary(summaries: List[str]) -> str:
    combined_text = "\n".join(summaries)
    doc_hash = hash_text(combined_text)
    if f"final_summary_{doc_hash}" in st.session_state:
        return st.session_state[f"final_summary_{doc_hash}"]
    doc = Document(text=combined_text)
    summary_index = SummaryIndex([doc])
    query_engine = summary_index.as_query_engine(response_mode="tree_summarize", use_async=True)
    response = await query_engine.aquery("Summarize and compare the key themes and findings of these documents")
    st.session_state[f"final_summary_{doc_hash}"] = response.response
    return response.response

def analyze_sentiment(text: str) -> dict:
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

def main():
    st.title("ML-Powered Research Assistant for Students")
    
    for key in ['summaries', 'final_summary', 'individual_summary', 'keywords', 'documents', 'query_engine', 'index']:
        if key not in st.session_state:
            st.session_state[key] = None
    
    uploaded_files = st.file_uploader("Upload PDF research papers", type=["pdf"], accept_multiple_files=True)
    
    if uploaded_files:
        documents = []
        for uploaded_file in uploaded_files:
            with open(f"temp_{uploaded_file.name}", "wb") as f:
                f.write(uploaded_file.getbuffer())
            text = extract_text_from_pdf(f"temp_{uploaded_file.name}")
            doc = Document(text=text, metadata={"doc_id": uploaded_file.name})
            documents.append(doc)
        
        doc_hashes = [hash_text(doc.text) for doc in documents]
        doc_hash_key = "".join(doc_hashes)
        if st.session_state.documents != documents or "index" not in st.session_state:
            st.session_state.documents = documents
            with st.spinner("Building retrieval index..."):
                splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
                nodes = splitter.get_nodes_from_documents(documents)
                st.session_state.index = VectorStoreIndex(nodes, embed_model=embed_model)
                st.session_state.query_engine = st.session_state.index.as_query_engine(
                    similarity_top_k=4, response_mode="compact"
                )
        
        if st.button("Generate Summaries"):
            with st.spinner("Generating summaries..."):
                st.session_state.summaries = asyncio.run(generate_all_summaries(documents))
                if len(st.session_state.summaries) == 1:
                    st.session_state.individual_summary = st.session_state.summaries[0]
                for i, summary in enumerate(st.session_state.summaries):
                    st.subheader(f"Summary for Document {i + 1}")
                    st.write(summary)
        
        if len(uploaded_files) > 1 and st.session_state.summaries and st.button("Generate Final Summary"):
            with st.spinner("Generating final summary..."):
                st.session_state.final_summary = asyncio.run(generate_final_summary(st.session_state.summaries))
                st.subheader("Final Summary Across All Documents")
                st.write(st.session_state.final_summary)
        
        if st.session_state.summaries and st.button("Analyze Sentiment of Summary"):
            with st.spinner("Analyzing sentiment..."):
                text_to_analyze = st.session_state.individual_summary if len(uploaded_files) == 1 else st.session_state.final_summary
                if not text_to_analyze:
                    st.error("No summary available for sentiment analysis.")
                    return
                sentiment_result = analyze_sentiment(text_to_analyze)
                st.subheader("Sentiment Analysis")
                st.write(f"Sentiment: {sentiment_result['label']} (Confidence: {sentiment_result['score']:.2f})")
        
        if st.session_state.documents and st.button("Extract Keywords"):
            with st.spinner("Extracting keywords..."):
                st.session_state.keywords = [extract_keywords(doc.text) for doc in st.session_state.documents]
                for i, keywords in enumerate(st.session_state.keywords):
                    st.subheader(f"Keywords for Document {i + 1}")
                    st.write(", ".join(keywords))
        
        if st.session_state.documents and st.button("Enable Q&A"):
            st.success("Q&A enabled! Ask a question below.")
        
        if st.session_state.query_engine:
            query = st.text_input("Ask a question about the papers:")
            if query:
                with st.spinner("Retrieving answer..."):
                    response = st.session_state.query_engine.query(query)
                    st.subheader("Answer")
                    st.write(response.response)

if __name__ == "__main__":
    main()