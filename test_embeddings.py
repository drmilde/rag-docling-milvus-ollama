import ollama

embeddings = ollama.embed(
    model="embeddinggemma",
    input = "Das ist ein Text"
)

print (embeddings.embeddings)
