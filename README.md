# Research Paper Reviewer – RAG System with LangChain & Ollama

A full-stack web application for intelligent research paper review. Upload PDFs and ask natural language questions—powered by Retrieval-Augmented Generation (RAG), LangChain, and local LLMs via Ollama.

## Features

- **Modern Web Interface:** React (Next.js) frontend with TypeScript  
- **Advanced RAG Pipeline:** Document chunking, semantic embeddings, and vector search  
- **Local LLM Integration:** Private, fast, and cost-free question answering with Ollama  
- **Contextual Q&A:** Accurate, source-cited answers from your uploaded papers  
- **Production-Ready:** Scalable, secure, and extensible architecture  

## Architecture

```
PDF Upload → Chunking & Embedding → Vector Store → Question → Retrieval → LLM Answer → Sources
```

- **Frontend:** Next.js (React, TypeScript, Tailwind CSS)  
- **Backend:** Flask API (Python), LangChain, FAISS, HuggingFace, Ollama  
- **LLM:** Local models via Ollama (e.g., Llama2, Mistral)  

## Quick Start

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd PaperWise
```

### 2. Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Ollama Setup

- [Download Ollama](https://ollama.ai/) and install for your OS  
- Pull a model (e.g., Llama2):

```bash
ollama pull llama2
ollama serve
```

### 4. Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

### 5. Run the Application

- **Backend:** `python app.py` (http://localhost:5000)  
- **Frontend:** `npm run dev` (http://localhost:3000)  

## Usage

1. **Upload a PDF** via the web interface.  
2. **Ask questions** about the paper’s content.  
3. **Receive answers** with cited sources and page numbers.  

## System Requirements

- **CPU:** 4+ cores  
- **RAM:** 16GB+ (Ollama models require memory)  
- **Storage:** 50GB+ SSD recommended  

## Key Technologies

- **Frontend:** Next.js, React, TypeScript, Tailwind CSS  
- **Backend:** Flask, LangChain, FAISS, HuggingFace, PyMuPDF  
- **LLM:** Ollama (Llama2, Mistral, etc.)  

## Roadmap & Future Enhancements

- Multi-document and cross-document Q&A  
- User authentication and document libraries  
- Cloud storage and distributed vector databases  
- Advanced analytics and export features  

## License

[MIT License](LICENSE)

---

**Contributions and feedback are welcome!**
