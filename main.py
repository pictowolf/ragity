from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import LanceDB
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

import lancedb
import os
from dotenv import load_dotenv

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

os.getenv("LANGCHAIN_API_KEY")
os.getenv("LANGCHAIN_TRACING_V2")
os.getenv("LANGCHAIN_ENDPOINT")
os.getenv("LANGCHAIN_PROJECT")

# start with uvicorn main:app --reload, make readme at some point

app = FastAPI(
    title="Ragity",
    description="RAG based LLM used to talk to internal documentation.",
    version="0.0.1",
    contact={
        "name": "PictoWolf",
        "email": "...."
    }
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def setup_embeddings(model_name, model_kwargs, encode_kwargs):
    return HuggingFaceBgeEmbeddings(model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)

def load_and_process_documents(loader_path):
    loader = PyPDFDirectoryLoader(loader_path)
    pdf_docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(pdf_docs)

def store_documents(documents, embeddings, db_path):
    db = lancedb.connect(db_path)
    table = db.create_table("docs", data=[
        {"vector": embeddings.embed_query("Hello World"), "text": "Hello World", "id": "1"}
    ], mode="overwrite")
    return LanceDB.from_documents(documents, embeddings, connection=table)

def query_documents(qa_system, query):
    return qa_system.invoke(query)

# Ingest documents endpoint
@app.post("/ingest")
async def ingest():
    model_name = "BAAI/bge-large-en-v1.5"
    model_kwargs = {'device': 'cuda'}
    encode_kwargs = {'normalize_embeddings': True}
    hf = setup_embeddings(model_name, model_kwargs, encode_kwargs)
    documents = load_and_process_documents("data/")
    store_documents(documents, hf, "./lancedb") 

# Stream chat
@app.post("/chat")
async def chat(content: str):
    model_name = "BAAI/bge-large-en-v1.5"
    model_kwargs = {'device': 'cuda'}
    encode_kwargs = {'normalize_embeddings': True}
    hf = setup_embeddings(model_name, model_kwargs, encode_kwargs)

    # Connect to existing database (assuming already exists)
    db = lancedb.connect("./lancedb")
    table = db.open_table("docs")
    docsearch = LanceDB(connection=table, embedding=hf)

    qa = RetrievalQA.from_chain_type(llm=Ollama(model="llama2"), chain_type="stuff", retriever=docsearch.as_retriever())
    result = query_documents(qa, content)
    return result["result"]