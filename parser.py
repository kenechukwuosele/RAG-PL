"""PDF ingestion utilities.

This module extracts text from papers, splits it into chunks, and loads the
paper corpus that gets indexed into the retrieval store.
"""

import fitz  # PyMuPDF — PDF parsing library
import os


def extract_text_from_pdf(pdf_path):
    """
    Extract raw text from a single PDF file.

    Args:
        pdf_path (str): Path to the PDF file

    Returns:
        str: Full extracted text from all pages
    """
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


def chunk_text(text, chunk_size=500):
    """
    Split text into fixed-size word chunks.

    Why chunk? Embedding models have token limits and work best
    on focused pieces of text, not entire documents.

    Args:
        text (str): Full document text
        chunk_size (int): Number of words per chunk (default 500)

    Returns:
        list[str]: List of text chunks
    """
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size
    return chunks


def load_all_papers(papers_dir="papers"):
    """
    Load, extract and chunk all PDFs in a directory.

    For each PDF:
        1. Extracts full text via PyMuPDF
        2. Splits text into chunks
        3. Stores filename, full text and chunks as a dict

    Args:
        papers_dir (str): Path to folder containing PDFs (default 'papers')

    Returns:
        list[dict]: Each dict has keys:
            - 'filename' (str): PDF filename
            - 'text'     (str): Full extracted text
            - 'chunks'   (list[str]): List of text chunks
    """
    papers = []
    for filename in os.listdir(papers_dir):
        if filename.endswith(".pdf"):
            path = os.path.join(papers_dir, filename)
            try:
                print(f"Extracting: {filename}")
                text = extract_text_from_pdf(path)
                chunks = chunk_text(text)
                papers.append({
                    "filename": filename,
                    "text": text,
                    "chunks": chunks
                })
            except Exception as e:
                print(f"FAILED: {filename} — {e}")

    print(f"\nLoaded {len(papers)} papers")
    return papers


if __name__ == "__main__":
    papers = load_all_papers()
    print("\n--- Preview ---")
    print(papers[0]["text"][:500])
    print(f"\nChunks in first paper: {len(papers[0]['chunks'])}")
    print(papers[0]["chunks"][0])