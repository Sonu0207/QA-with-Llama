import os
import re
import tempfile
from typing import List, Any
import streamlit as st
import torch
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.vectorstores import FAISS  # Changed from Chroma to FAISS
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge


# Configure Streamlit
st.set_page_config(page_title="📚 Document Q&A with GPT2", layout="wide")

# Session State Init
if "messages" not in st.session_state:
    st.session_state.messages = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None


# 🔧 Initialize LLM
def initialize_llm():
    model_id = "openai-community/gpt2"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        device_map="cpu"
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1
    )

    return HuggingFacePipeline(pipeline=pipe)


# 📂 Document loading
def load_document(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as temp:
        temp.write(file.getvalue())
        temp_path = temp.name

    ext = os.path.splitext(file.name)[1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(temp_path)
    elif ext == ".docx":
        loader = Docx2txtLoader(temp_path)
    elif ext == ".txt":
        loader = TextLoader(temp_path)
    else:
        os.unlink(temp_path)
        raise ValueError(f"Unsupported file type: {ext}")

    docs = loader.load()
    os.unlink(temp_path)
    return docs


# 🔍 Chunking + Embedding
def process_documents(documents: List[Any]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )
    chunks = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    # Changed from Chroma to FAISS
    vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    return retriever


# 🤖 Setup QA Chain
def setup_qa_chain(llm, retriever):
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )


# 🧠 Ask LLM
def generate_answer(qa_chain, question: str):
    result = qa_chain({"query": question})
    return {
        "answer": result["result"],
        "sources": result["source_documents"]
    }


# 💬 Chat History Source Formatter
def format_source(doc, i):
    content = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
    meta = doc.metadata
    out = f"**Source {i+1}**:\n\n{content}\n\n"
    if meta:
        out += "**Metadata:**\n" + "\n".join([f"- {k}: {v}" for k, v in meta.items()])
    return out


# 🧮 F1 Computation
def compute_f1(reference, prediction):
    ref_tokens = clean_and_tokenize(reference)
    pred_tokens = clean_and_tokenize(prediction)
    common = set(ref_tokens) & set(pred_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def clean_and_tokenize(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.split()


# 🚀 Streamlit App
def main():
    st.title("📚 Document Q&A with Evaluation")

    tab1, tab2 = st.tabs(["🔍 Ask Questions", "📊 Evaluate Model"])

    # Sidebar
    with st.sidebar:
        st.header("📂 Upload Documents")
        files = st.file_uploader("Upload PDFs, DOCX or TXT", type=["pdf", "docx", "txt"], accept_multiple_files=True)
        process = st.button("Process Documents")

    if files and process:
        all_docs = []
        for file in files:
            try:
                docs = load_document(file)
                all_docs.extend(docs)
                st.success(f"✅ Loaded {file.name}")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

        with st.spinner("🔍 Processing..."):
            retriever = process_documents(all_docs)
            llm = initialize_llm()
            st.session_state.qa_chain = setup_qa_chain(llm, retriever)
            st.success("✅ Ready! Use the tabs above to chat or evaluate.")

    # Tab 1 - Q&A
    with tab1:
        if st.session_state.qa_chain:
            st.header("🧠 Ask Questions")

            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    if "sources" in msg:
                        with st.expander("📄 Sources"):
                            for i, doc in enumerate(msg["sources"]):
                                st.markdown(format_source(doc, i))

            if question := st.chat_input("Ask a question about your documents..."):
                st.session_state.messages.append({"role": "user", "content": question})
                with st.chat_message("user"):
                    st.markdown(question)

                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = generate_answer(st.session_state.qa_chain, question)
                        st.markdown(response["answer"])
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response["answer"],
                            "sources": response["sources"]
                        })

                        with st.expander("📄 Sources"):
                            for i, doc in enumerate(response["sources"]):
                                st.markdown(format_source(doc, i))
        else:
            st.info("⬅️ Upload and process documents to get started.")

    # Tab 2 - Evaluation
    with tab2:
        st.header("📊 Evaluate Model Answer")

        if st.session_state.qa_chain:
            eval_question = st.text_input("Question for Evaluation")
            eval_reference = st.text_area("Ground Truth Answer (Reference)", height=100)
            evaluate_btn = st.button("Evaluate Answer")

            if evaluate_btn and eval_question and eval_reference:
                with st.spinner("Generating and Evaluating Answer..."):
                    response = generate_answer(st.session_state.qa_chain, eval_question)
                    prediction = response["answer"]
                    st.subheader("📌 Predicted Answer")
                    st.markdown(prediction)

                    # Compute Metrics
                    try:
                        rouge = Rouge()
                        rouge_scores = rouge.get_scores(prediction, eval_reference)[0]["rouge-l"]
                        bleu_score = sentence_bleu([eval_reference.split()], prediction.split())
                        f1 = compute_f1(eval_reference, prediction)

                        st.subheader("📈 Evaluation Metrics")
                        st.metric("BLEU Score", f"{bleu_score:.3f}")
                        st.metric("ROUGE-L F1", f"{rouge_scores['f']:.3f}")
                        st.metric("F1 Score", f"{f1:.3f}")
                    except Exception as e:
                        st.error(f"⚠️ Evaluation failed: {e}")
        else:
            st.warning("⚠️ Please process documents before evaluating.")


if __name__ == "__main__":
    main()