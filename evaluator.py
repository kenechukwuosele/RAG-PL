"""LLM-based response evaluation for the RAG pipeline.

This module asks the model to judge faithfulness, relevance, and completeness
of a generated answer against the retrieved context.
"""

from generator import client

def evaluate(query, docs, answer):
    """
    Use LLM to evaluate retrieval and answer quality.
    
    Args:
        query (str): Original question
        docs (list[str]): Retrieved chunks used for answer
        answer (str): Generated answer
    
    Returns:
        dict: scores and reasoning
    """
    context = "\n\n".join(docs)
    
    prompt = f"""Evaluate this RAG system response. Score each metric 1-5.

Question: {query}

Retrieved Context:
{context}

Generated Answer: {answer}

Score these metrics (1-5) and give one line reasoning each:
1. Faithfulness: Is the answer grounded in the context only?
2. Relevance: Does the context actually relate to the question?
3. Completeness: Does the answer cover the key points in context?

Respond in this exact format:
Faithfulness: X/5 | reasoning
Relevance: X/5 | reasoning
Completeness: X/5 | reasoning"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    
    result = response.choices[0].message.content.strip()
    return result