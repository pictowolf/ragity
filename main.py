from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders import S3DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import LanceDB
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

import lancedb
import os
from dotenv import load_dotenv
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    if os.getenv("S3_STORAGE") == 'True':
        logging.info("Loading from S3 Storage bucket: %s", os.getenv("S3_BUCKET"))
        loader = S3DirectoryLoader(
            bucket=os.getenv("S3_BUCKET"),
            endpoint_url=f'{os.getenv("S3_ENDPOINT")}',
            aws_access_key_id=os.getenv("S3_ACCESS_KEY"),
            aws_secret_access_key=os.getenv("S3_SECRET_KEY"),
            use_ssl=os.getenv("S3_SSL") == 'True'
        )
    else:
        logging.info("Loading from local storage path: %s", loader_path)
        loader = PyPDFDirectoryLoader(loader_path)

    pdf_docs = loader.load()
    logging.info("Amount of documents loaded: %s", len(pdf_docs))

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(pdf_docs)

def store_documents(documents, embeddings, db_path):
    db = lancedb.connect(db_path)
    logging.info("Recreating table in vector store.")

    table = db.create_table("docs", data=[
        {"vector": embeddings.embed_query("Hello World"), "text": "Hello World", "id": "1"}
    ], mode="overwrite")
    return LanceDB.from_documents(documents, embeddings, connection=table)

def query_documents(qa_system, query):
    return qa_system.invoke(query)

# Ingest documents endpoint
@app.post("/ingest")
async def ingest():
    logging.info("Starting Ingest process.")
    model_name = "BAAI/bge-large-en-v1.5"
    model_kwargs = {'device': 'cuda'}
    encode_kwargs = {'normalize_embeddings': True}

    hf = setup_embeddings(model_name, model_kwargs, encode_kwargs)
    documents = load_and_process_documents(os.getenv("LOCAL_PATH"))
    store_documents(documents, hf, os.getenv("LANCEDB_PATH"))
    logging.info("Finished Ingest process.")

# Stream chat - somehow broke this - no idea how - could be ollama 
@app.post("/chat")
async def chat(content: str):
    logging.info("Starting Chat process.")
    model_name = "BAAI/bge-large-en-v1.5"
    model_kwargs = {'device': 'cuda'}
    encode_kwargs = {'normalize_embeddings': True}

    hf = setup_embeddings(model_name, model_kwargs, encode_kwargs)
    logging.info("Connecting to database.")
    db = lancedb.connect(os.getenv("LANCEDB_PATH"))
    table = db.open_table("docs")

    logging.info("Searching documents.")
    docsearch = LanceDB(connection=table, embedding=hf)

    logging.info("Sending context to Ollama for processing.") # Dies here RiP code
    qa = RetrievalQA.from_chain_type(llm=Ollama(model="llama2"), chain_type="stuff", retriever=docsearch.as_retriever())

    result = query_documents(qa, content)
    logging.info("Finished Chat process.")
    return result["result"]