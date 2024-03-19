from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import TokenTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
import lancedb
import pyarrow as pa

embedding_model = OllamaEmbeddings()

db_path = "./lancedb"
db = lancedb.connect(db_path)

placeholder_embedding = embedding_model.embed_query("Hello World")
if placeholder_embedding is not None:
    placeholder_embedding_list = placeholder_embedding.tolist() if hasattr(placeholder_embedding, "tolist") else placeholder_embedding
    placeholder_data = [{
        "id": "placeholder",
        "text": "Placeholder text",
        "vector": placeholder_embedding_list,
    }]
    table = db.create_table(
        "my_table",
        data=placeholder_data,
        mode="overwrite",
    )

loader = PyPDFDirectoryLoader("data/")
docs = loader.load()
text_splitter = TokenTextSplitter()

for doc_index, doc in enumerate(docs):
    text_content = getattr(doc, 'page_content', None)
    if text_content:
        tokenized_doc = text_splitter.split_text(text_content)
        for chunk_index, chunk in enumerate(tokenized_doc):
            chunk_text = ' '.join(chunk)
            embedding = embedding_model.embed_query(chunk_text)
            if embedding:
                embedding_list = embedding.tolist() if hasattr(embedding, "tolist") else embedding
                try:
                    db["my_table"].add({
                        "id": f"{doc_index}_{chunk_index}",
                        "text": chunk_text,
                        "vector": embedding_list,
                    })
                except Exception as e:
                    print(f"Failed to insert data for chunk {chunk_index} of document {doc_index}: {e}")
            else:
                print(f"No embedding generated for chunk {chunk_index} of document {doc_index}.")
    else:
        print(f"Skipping a document {doc_index} due to missing text content.")

print("Embeddings inserted into LanceDB successfully.")
print(table.count_rows())