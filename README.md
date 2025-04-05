# 🤖 DocuMentor

![Python](https://img.shields.io/badge/Python-3.11-blue)
![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-purple)
![Ollama](https://img.shields.io/badge/LLM-Gemma3%3A4b-orange)
![Redis](https://img.shields.io/badge/Cache-Redis-red)
![License: MIT](https://img.shields.io/badge/License-MIT-green)

> **DocuMentor** is a multimodal RAG (Retrieval-Augmented Generation) assistant that allows users to ask questions about uploaded PDFs — extracting both **text and image content** and generating intelligent answers with LLMs.

---

## 📌 Features

- 📄 Upload and parse PDFs (text + image extraction)
- 🧠 Multimodal understanding using OCR on images
- 💬 Ask questions — receive concise, context-rich answers
- ⚡ Redis-based caching for fast repeated queries
- 🗂️ Vector similarity retrieval with ChromaDB
- 🧠 Answer generation using **Gemma 3 (4B)** LLM via **Ollama**

---

## 🧱 Tech Stack

| Component         | Library/Tool                          |
|------------------|----------------------------------------|
| PDF Parsing       | `unstructured`, `pdfminer.six`         |
| Image Text        | `Pillow`, `pytesseract`                |
| Text Splitting    | `langchain_text_splitters`             |
| Embeddings        | `OllamaEmbeddings (llama3.2)`          |
| Vector Store      | `chromadb`                             |
| Caching Layer     | `redis`                                |
| Language Model    | `Gemma3:4b via Ollama`                 |
| Interface         | `Streamlit`                            |

---
