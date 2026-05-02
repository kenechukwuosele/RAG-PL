"""Hybrid retrieval over the indexed paper corpus.

Dense retrieval uses Chroma embeddings, then BM25 reranks the candidate chunks
by lexical overlap with the user query.
"""

import chromadb
from embedder import embed_chunks
from rank_bm25 import BM25Okapi

# Connect to persistent local ChromaDB instance
client = chromadb.PersistentClient(path="db")
collection = client.get_or_create_collection("papers")


def build_bm25(documents):
    """
    Build a BM25 sparse index from a list of text documents.
    
    Args:
        documents (list[str]): List of text chunks
    
    Returns:
        BM25Okapi: Keyword search index
    """
    tokenized = [doc.lower().split() for doc in documents]
    return BM25Okapi(tokenized)


def index_papers(papers):
    """
    Embed and store all paper chunks into ChromaDB.
    Should only be run once — data persists in db/ folder.
    
    Args:
        papers (list[dict]): Output of load_all_papers()
                             Each dict has 'filename', 'text', 'chunks'
    """
    for paper in papers:
        chunks = paper["chunks"]
        embeddings = embed_chunks(chunks)
        ids = [f"{paper['filename']}_{i}" for i in range(len(chunks))]
        metadatas = [{"source": paper["filename"]} for _ in chunks]
        collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=chunks,
            metadatas=metadatas
        )
        print(f"Indexed: {paper['filename']}")


def retrieve(query, top_k=5):
    """
    Hybrid retrieval: dense (semantic) + sparse (BM25 keyword).
    
    Steps:
        1. Embed query → find top_k*2 semantically similar chunks (dense)
        2. Score those chunks by keyword match with query (BM25)
        3. Sort by BM25 score → return top_k
    
    Args:
        query (str): User's question
        top_k (int): Number of chunks to return (default 5)
    
    Returns:
        tuple: (docs, metas)
            docs  — list of top_k text chunks
            metas — list of dicts with 'source' (filename)
    """
    # Step 1 - Dense retrieval
    query_embedding = embed_chunks([query])[0].tolist()
    dense_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k * 2
    )
    dense_docs = dense_results["documents"][0]
    dense_metas = dense_results["metadatas"][0]

    # Step 2 - Sparse retrieval
    bm25 = build_bm25(dense_docs)
    scores = bm25.get_scores(query.lower().split())

    # Step 3 - Combine and rerank
    combined = sorted(
        zip(scores, dense_docs, dense_metas),
        key=lambda x: x[0],
        reverse=True
    )

    top = combined[:top_k]
    docs = [x[1] for x in top]
    metas = [x[2] for x in top]
    return docs, metas