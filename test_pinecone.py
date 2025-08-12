from pinecone import Client

pc = Client(api_key="pcsk_5TezHr_5FS18xfebbQxaGZwSRUELygH9RUq7Hnor9u7FdDDYoq4ztLAS2Fv2LC5W19Z4Me", environment="aped-4627-b74a")
print(pc.list_indexes())
