import torch
import ollama

# Check if GPU or MPS is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"CUDA GPU is enabled: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS GPU is enabled.")
else:
    raise OSError(
        "No GPU or MPS device found. Please check your environment and ensure GPU or MPS support is configured."
    )


############# DEFINE EMBEDDING

def emb_text(text):
    embeddings = ollama.embed(
        #model="embeddinggemma",
        #model="llama3.2",
        #model="gemma:2b",
        model="nomic-embed-text",
        input = text
    )   
    return embeddings.embeddings[0]

test_embedding = emb_text("This is a test")
embedding_dim = len(test_embedding)
print(embedding_dim)
print(test_embedding[:10])

###################### CONVERT INPUT DOCUMENTS

from docling_core.transforms.chunker import HierarchicalChunker
from docling.document_converter import DocumentConverter

converter = DocumentConverter()
chunker = HierarchicalChunker()

# Convert the input file to Docling Document
source = "https://milvus.io/docs/overview.md"
source = "https://www.hs-fulda.de/fileadmin/user_upload/FB_Angewandte_Informatik/Studium/Studiengaenge/Digitale_Medien/RZ_Flyer_BSc-DM_2021-web.pdf"
source = "/Users/cfc/Downloads/rag-data/LangGraphTutorial.html"
doc = converter.convert(source).document

# Perform hierarchical chunking
texts = [chunk.text for chunk in chunker.chunk(doc)]

##### SETTING UP DATABASE #########

from pymilvus import MilvusClient

milvus_client = MilvusClient(uri="./milvus_demo.db")
collection_name = "my_rag_collection"

if milvus_client.has_collection(collection_name):
    milvus_client.drop_collection(collection_name)

milvus_client.create_collection(
    collection_name=collection_name,
    dimension=embedding_dim,
    metric_type="IP",  # Inner product distance
    consistency_level="Strong",  # Supported values are (`"Strong"`, `"Session"`, `"Bounded"`, `"Eventually"`). See https://milvus.io/docs/consistency.md#Consistency-Level for more details.
)

#### INSERTING DATA ################

from tqdm import tqdm

data = []

for i, chunk in enumerate(tqdm(texts, desc="Processing chunks")):
    embedding = emb_text(chunk)
    data.append({"id": i, "vector": embedding, "text": chunk})

milvus_client.insert(collection_name=collection_name, data=data)


####### RAG SYSTEM aufbauen


def askMilvus(text):

    question = (
        #"What are the three deployment modes of Milvus, and what are their differences?"
        #"What Makes Milvus so Fast ?"
        #"What search algorithm does milvus support ?"
        text
    )
    #print(question)

    search_res = milvus_client.search(
        collection_name=collection_name,
        data=[emb_text(question)],
        limit=3,
        search_params={"metric_type": "IP", "params": {}},
        output_fields=["text"],
    )

    import json

    retrieved_lines_with_distances = [
        (res["entity"]["text"], res["distance"]) for res in search_res[0]
    ]
    #print(json.dumps(retrieved_lines_with_distances, indent=4))

    context = "\n".join(
        [line_with_distance[0] for line_with_distance in retrieved_lines_with_distances]
    )

    SYSTEM_PROMPT = """
    Human: You are an AI assistant. You are able to find answers to the questions from the contextual passage snippets provided.
    """

    USER_PROMPT = f"""
    Use the following pieces of information enclosed in <context> tags to provide an answer to the question enclosed in <question> tags.
    <context>
    {context}
    </context>
    <question>
    {question}
    </question>
    """

    #print (SYSTEM_PROMPT)
    #print (USER_PROMPT)

    ### GENERATE ANSWER

    from ollama import chat
    from ollama import ChatResponse

    response: ChatResponse = chat(model='gemma3', messages=[
    {
        'role': 'system',
        'content': SYSTEM_PROMPT,
    },
    {
        'role': 'user',
        'content': USER_PROMPT,
    },
    ])

    #print(response['message']['content'])
    return (response['message']['content'])
    # or access fields directly from the response object
    #print(response.message.content)


### CHAT with the document
while True:
    question = input("\nPose a question (/bye to exit) >>> ")
    if question == '/bye':
        break
    antwort = askMilvus(question)
    print(f"\n {antwort}")
