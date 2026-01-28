import streamlit as st
import torch
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms import HuggingFacePipeline

st.set_page_config(page_title="RAG Qwen PDF", layout="wide")
st.title("📚 Learning Assistant")

@st.cache_resource
def load_llm():
    model_id = "Qwen/Qwen2.5-0.5B-Instruct" 
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="cpu")
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=128)
    return HuggingFacePipeline(pipeline=pipe)

llm = load_llm()

uploaded_file = st.file_uploader("Upload your document", type="pdf")

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())
    
    loader = PyPDFLoader("temp.pdf")
    docs = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
    chunks = splitter.split_documents(docs)
    
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    st.success("Upload successful! You can now ask questions about the document.")
    
    question = st.text_input("Your question:")
    if question:
        retrieved_docs = vector_store.as_retriever(search_kwargs={"k": 5}).invoke(question)
        context = "\n".join(doc.page_content for doc in retrieved_docs)
        
        # Prompt 
        prompt = f"""<|im_start|>system
        You are a helpful assistant. Answer the question based on the context below concisely.
        If there is not enough information, say you don't know.<|im_end|>
        <|im_start|>user

        {context}

        Question: {question}<|im_end|>
        <|im_start|>assistant:"""

        response = llm.invoke(prompt)
        ans = response.split("<|im_start|>assistant:")[-1].split("<|im_end|>")[0].strip()
        st.markdown(f"**Trả lời:** {ans}")