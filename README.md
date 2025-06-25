# 🦙 QA‑with‑LLaMA

A Streamlit-based document Q&A app built using Meta’s LLaMA model (via API) and OpenAI GPT (e.g. GPT‑3.5). Upload a document and ask questions about it!

---

## 🔍 Features

- 📄 Upload any text-based file (PDF, DOCX, TXT, etc.)
- 💬 Ask natural-language questions about its content
- ✳ Context‑aware answers powered by GPT‑3.5 via OpenAI API
- 🌐 Web interface built with [Streamlit]([https://streamlit.io](https://app-with-llama-3gur3guqxdtb3iuf5jfdvg.streamlit.app/))
- ⚙️ Lightweight & easy to run locally

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit (`streamlit_app.py`)
- **Backend**: Python 3.x handling uploads, chunking, API calls
- **API**: OpenAI GPT‑3.5 via REST from Python
- **Requirements**: Listed in `requirements.txt`

---

## 🚀 Getting Started

### 1. Clone & Install

```bash
git clone https://github.com/Sonu0207/QA-with-Llama.git
cd QA-with-Llama
pip install -r requirements.txt
