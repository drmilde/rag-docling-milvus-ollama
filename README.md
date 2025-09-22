# rag-docling-milvus-ollama

An strictly local RAG-System (entry level) showing:

- how to use local ollama to generate the embeddings
- preprocessing the input document(s) using docling
- chunking the document and creating the embeddings
- create a collection in Milvus
- store the embeddings in this collection
- generate the embeddings for the given user question
- get the top 3 matching answers from the the database (Milvus)
- define a structured user prompt, injecting the information of the 3 answers and the user question
- use ollama to generate a natural language answer based on the given context
   
The code is based on the example by milvus.io on RAG with Milvus and Docling:

- https://milvus.io/docs/build_RAG_with_milvus_and_docling.md
