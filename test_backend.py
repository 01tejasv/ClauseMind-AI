from backend import pc, index

print("Available Pinecone indexes:")
for idx in pc.list_indexes():
    print(idx)

try:
    test_vector = [0.0]*1536
    index.upsert(vectors=[("test-id", test_vector, {"filename": "test.txt", "chunk": "Hello world!"})])
    print("Test vector upsert successful!")
except Exception as e:
    print("Upsert failed:", e)
