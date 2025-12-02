Mini RAG System
  This project is a minimal implementation of a Retrieval-Augmented Generation (RAG) system using Python and Sentence Transformers. It retrieves relevant documents using cosine similarity and generates simple answers based on the retrieved context.

How to Install
Install dependencies:
  pip install sentence-transformers numpy

How to Run
  python mini_rag.py
  You will then be able to ask questions in the command line.

What This Project Does
  Loads a small set of documents inside the script.
  Converts them into embeddings using SentenceTransformer.
  Takes a user query and embeds it.
  Calculates cosine similarity to rank the documents.
  Retrieves the top-K most relevant documents.
  Generates a simple answer using the retrieved documents.

Example
  Ask a question (or type 'exit'): What is the leave policy?
  [Retrieved documents]
  - doc_0: Employees get 12 paid leave days per year.
  - doc_2: You can work from home up to 2 days per week with manager approval.

  [Answer]
  Question: What is the leave policy?

  Based on the company documents, here is the answer:
  Employees get 12 paid leave days per year. You can work from home up to 2 days per week with manager approval.

Files Included
  mini_rag.py

Requirements
  Python 3.8+
  sentence-transformers
  numpy
