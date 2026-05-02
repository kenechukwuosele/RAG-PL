"""Cross-encoder reranking for retrieved chunks.

The reranker scores query-document pairs and keeps the strongest candidates
while rejecting obviously irrelevant results.
"""

import os
os.environ["TRANSFORMERS_OFFLINE"] = "0"

from sentence_transformers import CrossEncoder

model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query, docs, metas, top_k=5, threshold=0.0):
    """
    Rerank chunks. Returns empty lists if best score below threshold.
    
    Args:
        query (str): User question
        docs (list[str]): Retrieved chunks
        metas (list[dict]): Chunk metadata
        top_k (int): Chunks to return
        threshold (float): Min score to accept results
    
    Returns:
        tuple: (reranked_docs, reranked_metas)
    """
    pairs = [[query, doc] for doc in docs]
    scores = model.predict(pairs)

    combined = sorted(
        zip(scores, docs, metas),
        key=lambda x: x[0],
        reverse=True
    )

    # Reject if best score too low
    if combined[0][0] < threshold:
        return [], []

    top = combined[:top_k]
    return [x[1] for x in top], [x[2] for x in top]