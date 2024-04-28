from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import LanceDB
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import lancedb
import argparse

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

# Configuration and Setup
def main(args):
    model_name = "BAAI/bge-large-en-v1.5"
    model_kwargs = {'device': 'cuda'}
    encode_kwargs = {'normalize_embeddings': True}
    hf = setup_embeddings(model_name, model_kwargs, encode_kwargs)

    if args.ingest:
        documents = load_and_process_documents(args.data_directory)
        docsearch = store_documents(documents, hf, args.db_path)
    else:
        # Connect to existing database (assuming already exists)
        db = lancedb.connect(args.db_path)
        table = db.open_table("docs")
        docsearch = LanceDB(connection=table, embedding=hf)

    qa = RetrievalQA.from_chain_type(llm=Ollama(model="llama2"), chain_type="stuff", retriever=docsearch.as_retriever())

    # Interactive Query Loop
    while True:
        query = input("Enter your query (type 'exit' to stop): ")
        if query.lower() == 'exit':
            break
        result = query_documents(qa, query)
        print("Answer:", result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some documents.')
    parser.add_argument('--ingest', action='store_true', help='Ingest and process the PDF documents to build the database')
    parser.add_argument('data_directory', type=str, help='Directory where PDF documents are stored', default="data/", nargs='?')
    parser.add_argument('db_path', type=str, help='Path to the database file', default="./lancedb", nargs='?')
    args = parser.parse_args()
    main(args)