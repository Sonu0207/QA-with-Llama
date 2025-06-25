# 📄 Document Q&A System with LLaMA 3 & Local GPT-2 using LangChain and Streamlit

This project is an advanced interactive web application for **extractive and generative question answering over PDF documents**, leveraging state-of-the-art Large Language Models (LLMs) — Meta's LLaMA 3 via Groq API and a fallback local GPT-2 model — combined with vector search retrieval for contextual grounding.

---

## 🚀 Overview

This system enables users to:

- **Upload arbitrary PDF documents**
- **Automatically chunk and embed** document text for efficient semantic search
- **Perform retrieval-augmented generation (RAG)** using LangChain’s RetrievalQA pattern
- Interactively query content with **LLaMA 3 (Groq API)** or **local GPT-2** as the language model backend
- Optionally evaluate generated answers with **BLEU and ROUGE** metrics against user-provided reference answers
- Monitor detailed **latency and performance metrics** for both API and local inference

---

## 🔍 System Architecture & Components

| Component                 | Description                                                                                  |
|---------------------------|----------------------------------------------------------------------------------------------|
| **PDF Loader & Chunker**  | Loads PDF using `PyPDFLoader`, splits text with `RecursiveCharacterTextSplitter` (chunk_size=500, overlap=100) for coherent context windows. |
| **Vector Embeddings**     | Generates embeddings via HuggingFace's embedding models (`HuggingFaceEmbeddings`) for semantic retrieval. |
| **Vector Store**          | Stores embeddings in a `FAISS` index for fast nearest neighbor retrieval of relevant text chunks. |
| **Language Models (LLMs)**| - **Groq LLaMA 3:** Uses Groq API for chat completions with temperature control and token limits.<br>- **Local GPT-2:** Uses HuggingFace pipeline with timing instrumentation for offline fallback. |
| **RetrievalQA Chain**     | Combines retriever and LLM in LangChain's `RetrievalQA` chain with a custom prompt template emphasizing context-driven answers or fallback response "I don't know." |
| **Evaluation Module**     | Computes BLEU and ROUGE metrics using `nltk` and `rouge_score` against user-supplied reference answers for quality assessment. |
| **Streamlit UI**          | Interactive frontend for file upload, question input, model selection, and output display with real-time performance stats and evaluation results. |

---

## ⚙️ Installation & Setup

1. Clone this repository:

   ```bash
   git clone https://github.com/Sonu0207/QA-with-Llama.git
   cd QA-with-Llama
   pip install -r requirements.txt

## Link to app
[Streamlit](https://app-with-llama-3gur3guqxdtb3iuf5jfdvg.streamlit.app/)
