import json
from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import TokenTextSplitter
from langchain_community.embeddings import OllamaEmbeddings

loader = PyPDFDirectoryLoader("data/")
docs = loader.load()
text_splitter = TokenTextSplitter()
ollama = Ollama()
embedding_model = OllamaEmbeddings()

stored_embeddings = []

for doc in docs:
    text_content = getattr(doc, 'page_content', None)
    if text_content:
        tokenized_doc = text_splitter.split_text(text_content)
        for chunk in tokenized_doc:
            chunk_text = ' '.join(chunk)
            embedding = embedding_model.embed_query(chunk_text)
            if embedding:  # Check if embedding is not None or empty
                # Convert to list if necessary and append to stored_embeddings
                stored_embeddings.append(embedding.tolist() if hasattr(embedding, "tolist") else embedding)
            else:
                print("No embedding generated for chunk.")
    else:
        print(f"Skipping a document due to missing text content: {doc}")

#print the first few stored_embeddings to check
print("First few embeddings:", stored_embeddings[:3])

filepath = './embeddings.json'

try:
    with open(filepath, 'w') as f:
        json.dump(stored_embeddings, f)
    print(f"Embeddings stored successfully in {filepath}.")
except Exception as e:
    print(f"Failed to save embeddings at {filepath}: {e}")

print(len(stored_embeddings), len(stored_embeddings[0]))