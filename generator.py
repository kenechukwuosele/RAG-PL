"""LLM-based query expansion and answer generation.

The Groq client is used both to rewrite the user's question into alternate
search queries and to produce grounded answers from retrieved context.
"""

import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ"))


def generate_answer(query, retrieved_docs):
    context = "\n\n".join(retrieved_docs)
    prompt = f"""You are a research assistant. Answer the question using ONLY the context below.
If the answer isn't in the context, say "I don't have enough information."

Context:
{context}

Question: {query}

Answer:"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def expand_query(query):
    """
    Generate alternative versions of the query to improve retrieval.
    
    Args:
        query (str): Original user question
    
    Returns:
        list[str]: Original query + 3 alternatives
    """
    prompt = f"""Generate 3 alternative versions of this search query to improve document retrieval.
Return ONLY the 3 alternatives, one per line, no numbering, no explanation.

Query: {query}"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    alternatives = response.choices[0].message.content.strip().split("\n")
    alternatives = [q.strip() for q in alternatives if q.strip()]
    return [query] + alternatives[:3]