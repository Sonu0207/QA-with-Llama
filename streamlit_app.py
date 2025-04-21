import streamlit as st
import time  # <-- NEW
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
from langchain.llms.base import LLM
import tempfile
import requests
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt_tab')


# ====== HARDCODED API KEY ======
GROQ_API_KEY = "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"
# ===============================

# Try native ChatGroq
try:
    from langchain_groq import ChatGroq
    GROQ_SUPPORTED = True
except ImportError:
    GROQ_SUPPORTED = False

# Manual Groq wrapper
class GroqLLM(LLM):
    model_name: str = "llama3-8b-8192"
    temperature: float = 0.7
    max_tokens: int = 500
    last_response_time: float = 0.0  # <-- Added

    def _call(self, prompt: str, stop=None) -> str:
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }

        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }

        start_time = time.time()
        response = requests.post("https://api.groq.com/openai/v1/chat/completions", json=data, headers=headers)
        self.last_response_time = time.time() - start_time

        if response.status_code != 200:
            raise Exception(f"Groq API error: {response.status_code} - {response.text}")

        return response.json()["choices"][0]["message"]["content"]

    @property
    def _llm_type(self) -> str:
        return "groq"

# Groq initializer
def initialize_groq_llm(model_name="llama3-8b-8192"):
    if GROQ_SUPPORTED:
        return ChatGroq(
            model_name=model_name,
            temperature=0.7,
            max_tokens=500,
            api_key=GROQ_API_KEY
        )
    else:
        return GroqLLM(model_name=model_name)

# Local fallback LLM
class TimedHuggingFacePipeline(HuggingFacePipeline):  # <-- Add timing to local LLM
    def __call__(self, *args, **kwargs):
        start = time.time()
        output = super().__call__(*args, **kwargs)
        self.response_time = time.time() - start
        return output

def initialize_local_llm():
    hf_pipeline = pipeline("text-generation", model="gpt2", max_new_tokens=300)
    timed_pipeline = TimedHuggingFacePipeline(pipeline=hf_pipeline)
    return timed_pipeline

# PDF processing
def process_pdf_to_vectorstore(pdf_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_file.read())
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_documents(splits, embeddings)

    return vectorstore

# QA Chain builder
def build_qa_chain(llm, vectorstore):
    retriever = vectorstore.as_retriever()

    prompt_template = PromptTemplate(
        template="""Answer the question based only on the context below. 
If you don't know the answer, say you don't know.

Context:
{context}

Question:
{question}""",
        input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt_template}
    )

    return chain

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("📄 Document Q&A with LLaMA or Local GPT-2")
st.markdown("Upload a PDF and ask questions using Meta LLaMA or local GPT-2.")

pdf_file = st.file_uploader("Upload your PDF", type=["pdf"])
question = st.text_input("Ask a question")
reference_answer = st.text_area("Optional: Reference Answer (for BLEU/ROUGE)")
model_choice = st.radio("Choose your model:", ["LLaMA 3", "Local GPT-2"])

if st.button("Run Q&A") and pdf_file and question:
    st.info("Processing...")

    vectorstore = process_pdf_to_vectorstore(pdf_file)

    llm = None
    if model_choice == "LLaMA 3":
        try:
            llm = initialize_groq_llm()
        except Exception as e:
            st.error(f"Groq LLaMA failed: {e}")
            st.stop()
    else:
        llm = initialize_local_llm()

    # Track performance time
    start_time = time.time()
    qa_chain = build_qa_chain(llm, vectorstore)
    answer = qa_chain.run(question)
    total_time = time.time() - start_time

    # Output
    st.success("Answer:")
    st.write(answer)

    # Metrics
    st.markdown("### 📊 Performance Metrics")
    st.write(f"**Model Used:** {model_choice}")
    st.write(f"**Total Response Time:** {total_time:.2f} seconds")

    if isinstance(llm, GroqLLM):
        st.write(f"**Groq API Time:** {llm.last_response_time:.2f} seconds")
    elif isinstance(llm, TimedHuggingFacePipeline):
        st.write(f"**Local Generation Time:** {llm.response_time:.2f} seconds")

    # BLEU and ROUGE (if reference provided)
    if reference_answer.strip():
        st.markdown("### 🧪 Evaluation Metrics (BLEU & ROUGE)")
        try:
            reference = word_tokenize(reference_answer.strip().lower())
            candidate = word_tokenize(answer.strip().lower())

            # BLEU
            smoothie = SmoothingFunction().method4
            bleu_score = sentence_bleu([reference], candidate, smoothing_function=smoothie)
            st.write(f"**BLEU Score:** {bleu_score:.4f}")

            # ROUGE
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
            rouge_scores = scorer.score(reference_answer, answer)
            st.write(f"**ROUGE-1 F1:** {rouge_scores['rouge1'].fmeasure:.4f}")
            st.write(f"**ROUGE-L F1:** {rouge_scores['rougeL'].fmeasure:.4f}")
        except Exception as e:
            st.warning(f"⚠️ Evaluation failed: {e}")
