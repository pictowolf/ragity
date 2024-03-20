import re
import pickle
import requests
import zipfile
from pathlib import Path

from langchain_community.document_loaders import UnstructuredHTMLLoader, PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import LanceDB
from langchain.chains import RetrievalQA
# from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
# from langchain_community.llms import OpenAI
# from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

model_name = "BAAI/bge-large-en-v1.5"
model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity
hf = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

#Source Data
loader = PyPDFDirectoryLoader("data/")
pdf_docs = loader.load()

# Parse Data
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
documents = text_splitter.split_documents(pdf_docs)

# Embed Data
embeddings = hf
# embeddings = OpenAIEmbeddings() # OpenAI embedding is so much better, bad results via ollama for embedding

# Store Data
import lancedb

db_path = "./lancedb"
db = lancedb.connect(db_path)

table = db.create_table("pandas_docs", data=[
    {"vector": embeddings.embed_query("Hello World"), "text": "Hello World", "id": "1"}
], mode="overwrite")
docsearch = LanceDB.from_documents(documents, embeddings, connection=table)

qa = RetrievalQA.from_chain_type(llm=Ollama(model="llama2"), chain_type="stuff", retriever=docsearch.as_retriever())
# qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever())

query = "What are the 9 different FATCA account statuses?"
print(qa.invoke(query))