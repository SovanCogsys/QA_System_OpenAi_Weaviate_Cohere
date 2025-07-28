# QA_System_OpenAi_Weaviate_Cohere
This system implements a Retrieval-Augmented Generation (RAG) pipeline using proprietary APIs from OpenAI, Weaviate, and Cohere. It is designed for accurate and interpretable educational question answering using textbook-aligned data.

# Core Features
Multimodal PDF-to-Text Pipeline: Converts textbook PDFs into clean, structured text using pdf2image + GPT-4o Vision. Extracts content, equations (in natural language), headings, tables, and image captions.

Semantic Chunking: Splits pages into coherent content chunks based on visual headings and stores them with rich metadata (chapter, title, section, page number).

Weaviate Vector Store:

Stores all semantic chunks as objects in Weaviate collections.

Uses text-embedding-3-large for content vectorization.

Hybrid search (BM25 + vector) with reranking using cohere-rerank-english-v3.0.

Custom RAG QA Interface:

User inputs subject and query.

Relevant chunks retrieved + reranked.

Final answer generated via GPT-4o using a custom prompt template.

Cited sources (chapter and page) also displayed.

💡 Use Cases
Automated answering of textbook-aligned questions

Educational chatbots for NEET/JEE exam preparation

Building subject-specific QA datasets

Curriculum-based tutoring applications

📦 Stack
LLM + Embedding: OpenAI GPT-4o, text-embedding-3-large

Database: Weaviate (cloud instance)

Reranking: Cohere rerank-english-v3.0

OCR + Chunking: pdf2image, openai-vision, regex-based heading tracker

Frameworks: LangChain, Python
