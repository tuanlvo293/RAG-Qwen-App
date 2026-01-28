import streamlit as st
import torch
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms import HuggingFacePipeline

st.set_page_config(page_title="RAG Qwen PDF", layout="wide")
st.title("📚 Hệ thống RAG - PhD Assistant")

@st.cache_resource
def load_llm():
    # Sử dụng bản 0.5B để nhẹ máy
    model_id = "Qwen/Qwen2.5-0.5B-Instruct" 
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="cpu")
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=128)
    return HuggingFacePipeline(pipeline=pipe)

llm = load_llm()

uploaded_file = st.file_uploader("Tải lên tài liệu PDF của bạn", type="pdf")

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())
    
    loader = PyPDFLoader("temp.pdf")
    docs = loader.load()
    
    # Chia nhỏ văn bản theo đúng thông số Colab của bạn
    splitter = RecursiveCharacterTextSplitter(chunk_size=588, chunk_overlap=108)
    chunks = splitter.split_documents(docs)
    
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    st.success("Tài liệu đã được nạp thành công!")
    
    question = st.text_input("Câu hỏi của bạn:")
    if question:
        retrieved_docs = vector_store.as_retriever(search_kwargs={"k": 4}).invoke(question)
        context = "\n".join(doc.page_content for doc in retrieved_docs)
        
        # Prompt chuẩn từ mã nguồn bạn cung cấp
        prompt = f"""<|im_start|>system
Bạn là một trợ lý hữu ích. Trả lời câu hỏi dựa trên ngữ cảnh dưới đây một cách ngắn gọn.
Nếu không thấy đủ thông tin, hãy nói bạn không biết. <|im_end|>
<|im_start|>user
Ngữ cảnh:
{context}

Câu hỏi: {question}<|im_end|>
<|im_start|>assistant:"""

        response = llm.invoke(prompt)
        # Tách đáp án dựa trên cấu trúc Qwen
        ans = response.split("<|im_start|>assistant:")[-1].split("<|im_end|>")[0].strip()
        st.markdown(f"**Trả lời:** {ans}")