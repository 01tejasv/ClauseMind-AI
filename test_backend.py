from backend import pc, pc as index  # use same instance for testing

# List all indexes
print("Available Pinecone indexes:")
for idx in pc.list_indexes():
    print(idx)

# Optional: test a simple upsert
try:
    test_vector = [0.0]*1536  # dummy embedding
    index.upsert(vectors=[("test-id", test_vector, {"filename": "test.txt", "chunk": "Hello world!"})])
    print("Test vector upsert successful!")
except Exception as e:
    print("Upsert failed:", e)
