from pathlib import Path
from groq import Groq
import os
from dotenv import load_dotenv
import chromadb

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

DOCS_DIR = Path("documents")

documents = []

for file in DOCS_DIR.glob("*.txt"):
    text = file.read_text(encoding="utf-8")
    documents.append({
        "source": file.name,
        "text": text
    })

for doc in documents:
    print("\nSOURCE:", doc["source"])
    print(doc["text"])

def chunk_text(text, chunk_size=80, overlap=20):
    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))

        if end >= len(words):
            break

        start = end - overlap

    return chunks


sample = " ".join(["policy"] * 200)
chunks = chunk_text(sample)
print("Number of chunks:", len(chunks))
print("First chunk:", chunks[0])


def get_embedding(text):
    response = client.embeddings.create(
        model="YOUR_EMBEDDING_MODEL",
        input=text
    )
    return response.data[0].embedding


vector = get_embedding("Hotel reimbursement limit")
print("Vector length:", len(vector))
print("First five values:", vector[:5])




chroma_client = chromadb.PersistentClient(path="./chroma_db")

collection = chroma_client.get_or_create_collection(
    name="hr_policies"
)

ids = []
texts = []
metadatas = []
embeddings = []

counter = 0

for doc in documents:
    for chunk in chunk_text(doc["text"]):
        ids.append(f"chunk-{counter}")
        texts.append(chunk)
        metadatas.append({"source": doc["source"]})
        embeddings.append(get_embedding(chunk))
        counter += 1

collection.add(
    ids=ids,
    documents=texts,
    metadatas=metadatas,
    embeddings=embeddings
)
print("Stored chunks:", collection.count())

def retrieve(question, top_k=3):
    question_embedding = get_embedding(question)

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k
    )

    return results

results = retrieve("What is the hotel reimbursement limit?")

for doc_text, metadata in zip(
    results["documents"][0],
    results["metadatas"][0]
):
    print("\nSOURCE:", metadata["source"])
    print(doc_text)

def build_context(results):
    parts = []

    for doc_text, metadata in zip(
        results["documents"][0],
        results["metadatas"][0]
    ):
        parts.append(
            f"Source: {metadata['source']}\n"
            f"Content: {doc_text}"
        )

    return "\n\n".join(parts)

question = "What is the hotel reimbursement limit?"
results = retrieve(question)

context = build_context(results)

prompt = f"""
Answer the user's question using only the policy context below.
If the answer is not present, say that the policy context does not provide the answer.
Mention the source document in the answer.

POLICY CONTEXT:
{context}

USER QUESTION:
{question}
"""

response = client.responses.create(
    model="YOUR_MODEL_NAME",
    input=prompt
)

print(response.output_text)

def ask_rag(question):
    results = retrieve(question, top_k=3)
    context = build_context(results)

    prompt = f"""
    Answer using only the supplied policy context.
    If the context does not contain the answer, say so.
    Include the source document.

    CONTEXT:
    {context}

    QUESTION:
    {question}
    """

    response = client.responses.create(
        model="YOUR_MODEL_NAME",
        input=prompt
    )

    return response.output_text

print(ask_rag("How many annual leave days do employees receive?"))
print(ask_rag("What is the hotel reimbursement limit?"))
print(ask_rag("Can I claim an airport taxi?"))
print(ask_rag("What is the company's office dress code?"))

for doc_text, metadata in zip(
    results["documents"][0],
    results["metadatas"][0]
):
    print("Source:", metadata["source"])


from pypdf import PdfReader

reader = PdfReader("documents/company_policy.pdf")

text = ""
for page_number, page in enumerate(reader.pages, start=1):
    page_text = page.extract_text() or ""
    text += f"\n[Page {page_number}]\n{page_text}"

print(text[:3000])


while True:
    question = input("\nAsk a policy question (or type exit): ")

    if question.lower() == "exit":
        break

    print("\nAnswer:")
    print(ask_rag(question))