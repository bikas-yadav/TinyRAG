from sentence_transformers import SentenceTransformer
import numpy as np

# 1. Our tiny document "knowledge base"
documents = [
    "Employees get 12 paid leave days per year.",
    "The office is open from Monday to Friday, 9 AM to 6 PM.",
    "You can work from home up to 2 days per week with manager approval.",
    "Health insurance is provided after 3 months of joining the company.",
]

# Give IDs or titles so we can refer back
doc_ids = [f"doc_{i}" for i in range(len(documents))]

# 2. Load an embedding model (turns text -> vectors)
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# 3. Create embeddings for all documents
print("Creating document embeddings...")
doc_embeddings = model.encode(documents, convert_to_numpy=True)

# Simple cosine similarity function
def cosine_similarity(a, b):
    a = a / (np.linalg.norm(a) + 1e-10)
    b = b / (np.linalg.norm(b) + 1e-10)
    return np.dot(a, b)

def retrieve_top_k(query, k=2):
    """
    Given a query string:
      - embed the query
      - compute similarity with all documents
      - return top k most similar documents
    """
    query_embedding = model.encode([query], convert_to_numpy=True)[0]
    scores = [cosine_similarity(query_embedding, emb) for emb in doc_embeddings]
    # sort by score descending
    ranked = sorted(
        zip(doc_ids, documents, scores),
        key=lambda x: x[2],
        reverse=True
    )
    return ranked[:k]

def simple_generate_answer(query, retrieved_docs):
    """
    For this mini demo, we simulate 'generation':
    - We just combine the query and retrieved docs into a nice answer.
    """
    context_texts = [doc for _, doc, _ in retrieved_docs]
    context_joined = " ".join(context_texts)

    answer = (
        f"Question: {query}\n\n"
        f"Based on the company documents, here is the answer:\n"
        f"{context_joined}\n"
    )
    return answer

if __name__ == "__main__":
    print("Mini RAG system ready!")
    while True:
        query = input("\nAsk a question (or type 'exit'): ")
        if query.lower().strip() in ["exit", "quit"]:
            break

        # 4. RETRIEVAL: get top relevant docs
        top_docs = retrieve_top_k(query, k=2)
        print("\n[Retrieved documents]")
        for doc_id, doc_text, score in top_docs:
            print(f"- {doc_id} (score={score:.3f}): {doc_text}")

        # 5. GENERATION: build an answer using retrieved docs
        answer = simple_generate_answer(query, top_docs)
        print("\n[Answer]")
        print(answer)
