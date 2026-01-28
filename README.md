# 🤖 basicRAG: Qwen-Powered PDF Chatbot

This project implements a fundamental **RAG (Retrieval-Augmented Generation)** system, allowing users to upload PDF documents and engage in conversational Q&A based on the content. The system is optimized for efficiency in CPU-only environments.

---

## 🚀 Key Features

- **Document Processing**: Automated text extraction from uploaded PDF files  
- **Semantic Search**: Powered by FAISS for high-speed retrieval of relevant document segments  
- **Intelligent Response**: Integrated with the **Qwen2.5-0.5B** Large Language Model (LLM) for context-aware answers  
- **User-Friendly Interface**: Built with Streamlit for seamless browser-based interaction  

---

## 🛠 Technical Specifications

The system utilizes optimized parameters to balance performance and resource usage:

- **Embedding Model**: `BAAI/bge-m3`  
- **Chunk Size**: `588`  
- **Chunk Overlap**: `108`  
- **LLM**: `Qwen/Qwen2.5-0.5B-Instruct`  

---

## 📦 Installation & Usage

### 1. Clone the repository

```bash
git clone https://github.com/tuanlvo293/RAG-Qwen-App.git
cd RAG-Qwen-App
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the application locally

```bash
streamlit run src/app.py
```

---

## 📂 Project Structure

- `src/app.py` — Main source code handling RAG logic and UI  
- `requirements.txt` — Python dependencies (`langchain`, `transformers`, `faiss-cpu`, etc.)  
- `Dockerfile` — Environment configuration for deployment on Hugging Face Spaces  
- `.gitignore` — Excludes system junk (`.DS_Store`) and temporary files (`temp.pdf`)  

---

## 🎓 About the Author

**Võ Long Tuấn**

- Research Interests: XAI, RAG, missing data  

---

*Developed for rapid retrieval of research papers and academic textbooks during PhD research.*
