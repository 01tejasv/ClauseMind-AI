import openai
import pinecone
from uuid import uuid4

# -------------------------
# 1. Set your API keys
# -------------------------
openai.api_key = "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"

pinecone.init(
    api_key="pcsk_5TezHr_5FS18xfebbQxaGZwSRUELygH9RUq7Hnor9u7FdDDYoq4ztLAS2Fv2LC5W19Z4Me",
    environment="aped-4627-b74a"
)

# -------------------------
# 2. Index setup
# -------------------------
index_name = "clause-mind-index"

# Create index if it does not exist
if index_name not in pinecone.list_indexes():
    pinecone.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine"
    )

# Connect to index
index = pinecone.Index(index_name)

# -------------------------
# 3. Function to process & index documents
# -------------------------
def process_and_index(content: str, filename: str):
    # Split content into 1000-char chunks
    chunks = [content[i:i+1000] for i in range(0, len(content), 1000)]

    for chunk in chunks:
        embedding = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=chunk
        )["data"][0]["embedding"]

        metadata = {"filename": filename, "chunk": chunk}
        index.upsert(vectors=[(str(uuid4()), embedding, metadata)])

    return {"status": "success", "chunks_indexed": len(chunks)}
