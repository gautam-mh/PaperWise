import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import glob



class ResearchPaperRAG:
    def __init__(self):
        """Initialize the RAG system"""
        self.embeddings = None
        self.vectorstore = None
        self.qa_chain = None
        self.documents = []
        
    def load_documents(self, pdf_folder="data"):
        """
        Load all PDF documents from the specified folder
        """
        print(f"📁 Loading documents from {pdf_folder}...")
        
        # Find all PDF files in the folder
        pdf_files = glob.glob(os.path.join(pdf_folder, "*.pdf"))
        print(f"Found {len(pdf_files)} PDF files: {[os.path.basename(f) for f in pdf_files]}")
        
        all_documents = []
        
        for pdf_file in pdf_files:
            print(f"📄 Loading {os.path.basename(pdf_file)}...")
            
            # Load PDF using LangChain's PyMuPDFLoader
            loader = PyMuPDFLoader(pdf_file)
            documents = loader.load()
            
            # Add filename to metadata for each document
            for doc in documents:
                doc.metadata['filename'] = os.path.basename(pdf_file)
                doc.metadata['source_file'] = pdf_file
            
            all_documents.extend(documents)
            print(f"  ✅ Loaded {len(documents)} pages")
        
        self.documents = all_documents
        print(f"📚 Total documents loaded: {len(all_documents)} pages")
        return all_documents
    
    def chunk_documents(self, chunk_size=1000, chunk_overlap=200):
        """
        Split documents into smaller chunks for better retrieval
        """
        print(f"✂️ Chunking documents (size={chunk_size}, overlap={chunk_overlap})...")
        
        # Initialize the text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]  # Try to split on paragraphs first
        )
        
        # Split all documents
        chunks = text_splitter.split_documents(self.documents)
        
        # Add chunk information to metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata['chunk_id'] = i
            chunk.metadata['chunk_size'] = len(chunk.page_content)
        
        print(f"📝 Created {len(chunks)} chunks")
        
        # Show sample chunk
        if chunks:
            sample_chunk = chunks[0]
            print(f"\n📋 Sample chunk:")
            print(f"  File: {sample_chunk.metadata.get('filename', 'Unknown')}")
            print(f"  Page: {sample_chunk.metadata.get('page', 'Unknown')}")
            print(f"  Size: {len(sample_chunk.page_content)} characters")
            print(f"  Content preview: {sample_chunk.page_content[:200]}...")
        
        return chunks

    
    def create_embeddings_and_vectorstore(self, chunks):
        """
        Create embeddings for chunks and build vector store
        """
        print(f"🧠 Creating embeddings for {len(chunks)} chunks...")
        
        # Initialize HuggingFace embeddings (free, local)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",  # Same model as your original system!
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Create vector store from documents
        print("🔍 Building FAISS vector store...")
        self.vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding=self.embeddings
        )
        
        print("✅ Vector store created successfully!")
        return self.vectorstore
    
    def save_vectorstore(self, save_path="vectorstore_langchain"):
        """
        Save the vector store to disk
        """
        if self.vectorstore is None:
            print("❌ No vector store to save. Create embeddings first.")
            return
        
        print(f"💾 Saving vector store to {save_path}...")
        self.vectorstore.save_local(save_path)
        print("✅ Vector store saved!")
    
    def load_vectorstore(self, load_path="vectorstore_langchain"):
        """
        Load vector store from disk
        """
        print(f"📂 Loading vector store from {load_path}...")
        
        # Initialize embeddings first
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Load the vector store
        self.vectorstore = FAISS.load_local(
            load_path, 
            self.embeddings,
            allow_dangerous_deserialization=True  # Required for FAISS loading
        )
        print("✅ Vector store loaded!")
        return self.vectorstore
    
    def test_similarity_search(self, query, k=3):
        """
        Test similarity search with a query
        """
        if self.vectorstore is None:
            print("❌ No vector store available. Create embeddings first.")
            return
        
        print(f"🔍 Searching for: '{query}'")
        print(f"📊 Retrieving top {k} similar chunks...")
        
        # Perform similarity search
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        
        print(f"\n📋 Search Results:")
        for i, (doc, distance) in enumerate(results, 1):
            similarity_percent = max(0, (2.0 - distance) / 2.0 * 100)  # Rough conversion
            print(f"\n--- Result {i} (Distance: {distance:.4f}, ~{similarity_percent:.1f}% similar) ---")
            print(f"File: {doc.metadata.get('filename', 'Unknown')}")
            print(f"Page: {doc.metadata.get('page', 'Unknown')}")
            print(f"Content: {doc.page_content[:300]}...")
        
        return results

    
    def setup_qa_chain(self, model_name="llama2"):
        """
        Set up the Question-Answering chain using Ollama LLM
        
        This creates the complete RAG pipeline:
        Query → Retrieve Context → Generate Answer
        
        Args:
            model_name (str): Name of the Ollama model to use
        """
        if self.vectorstore is None:
            print("❌ No vector store available. Create embeddings first.")
            return
        
        print(f"🤖 Setting up QA chain with Ollama model: {model_name}")
        
        # Initialize Ollama LLM
        llm = OllamaLLM(
            model=model_name,
            temperature=0.1,  # Low temperature for more focused, factual answers
            # base_url="http://localhost:11434"  # Default Ollama URL
        )
        
        # Create custom prompt template for RAG
        prompt_template = """
        You are a helpful research assistant. Use the following pieces of context from research papers to answer the question at the end. 

        IMPORTANT INSTRUCTIONS:
        - Only use information from the provided context
        - If you don't know the answer based on the context, say "I don't have enough information in the provided context to answer this question."
        - Be specific and cite which paper/section your information comes from when possible
        - Provide detailed, well-structured answers

        Context from research papers:
        {context}

        Question: {question}

        Answer: 
        """
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create the QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",  # "stuff" means put all retrieved docs into one prompt
            retriever=self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}  # Retrieve top 4 most relevant chunks
            ),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True  # Include source documents in response
        )
        
        print("✅ QA chain setup complete!")
        return self.qa_chain

    
    def ask_question(self, question):
        """
        Ask a question and get an answer using RAG
        
        This is the main method users will call:
        1. Takes user question
        2. Retrieves relevant chunks from vector store
        3. Sends chunks + question to LLM
        4. Returns generated answer with sources
        
        Args:
            question (str): The question to ask about the documents
            
        Returns:
            dict: Contains 'result' (answer) and 'source_documents' (sources)
        """
        if self.qa_chain is None:
            print("❌ QA chain not set up. Run setup_qa_chain() first.")
            return
        
        print(f"❓ Question: {question}")
        print("🔍 Searching for relevant information...")
        
        # Get answer from the QA chain
        # This does: retrieve → augment prompt → generate answer
        response = self.qa_chain.invoke({"query": question})
        
        print(f"\n🤖 Answer:")
        print(f"{response['result']}")
        
        # Show source documents for transparency
        print(f"\n📚 Sources Used:")
        for i, doc in enumerate(response['source_documents'], 1):
            print(f"\n--- Source {i} ---")
            print(f"📄 File: {doc.metadata.get('filename', 'Unknown')}")
            print(f"📃 Page: {doc.metadata.get('page', 'Unknown')}")
            print(f"📝 Content: {doc.page_content[:200]}...")
        
        return response


    def quick_load_and_ask(self, question, vectorstore_path="vectorstore_langchain"):
        """
        Convenience method: Load saved vectorstore and ask a question
        Perfect for your final user-upload system!
        
        Args:
            question (str): Question to ask
            vectorstore_path (str): Path to saved vectorstore
        """
        try:
            # Load existing vectorstore
            self.load_vectorstore(vectorstore_path)
            
            # Setup QA chain
            self.setup_qa_chain()
            
            # Ask question
            return self.ask_question(question)
            
        except Exception as e:
            print(f"❌ Error: {e}")
            print("Make sure you have:")
            print("1. Created and saved a vectorstore first")
            print("2. Ollama is running with llama2 model")
            return None

if __name__ == "__main__":
    # Initialize RAG system
    rag = ResearchPaperRAG()
    
    # Load documents
    documents = rag.load_documents()
    
    # Chunk documents
    chunks = rag.chunk_documents()
    
    # Show statistics
    print(f"\n📊 Statistics:")
    print(f"  Total pages: {len(documents)}")
    print(f"  Total chunks: {len(chunks)}")
    
    # Show chunks per file
    file_chunks = {}
    for chunk in chunks:
        filename = chunk.metadata.get('filename', 'Unknown')
        file_chunks[filename] = file_chunks.get(filename, 0) + 1
    
    print(f"  Chunks per file:")
    for filename, count in file_chunks.items():
        print(f"    {filename}: {count} chunks")
    
    # Create embeddings and vector store
    print(f"\n" + "="*50)
    vectorstore = rag.create_embeddings_and_vectorstore(chunks)
    
    # Save vector store for future use
    rag.save_vectorstore()
    
    # Test similarity search
    print(f"\n" + "="*50)
    print("🧪 Testing Similarity Search")
    
    test_queries = [
        "What are the nine planetary boundaries?",
        "How do neural networks learn word representations?"
    ]
    
    for query in test_queries:
        print(f"\n" + "-"*30)
        rag.test_similarity_search(query, k=2)
    
    # Set up RAG QA chain
    print(f"\n" + "="*60)
    print("🤖 Setting up RAG Question-Answering System")
    
    rag.setup_qa_chain(model_name="llama2")
    
    # Test RAG Q&A
    print(f"\n" + "="*60)
    print("🎯 Testing Complete RAG Pipeline")
    
    rag_questions = [
        "What are the nine planetary boundaries and why are they important?",
        "How do neural networks create word representations that capture semantic meaning?",
        "What are the main conclusions and implications of these research papers?"
    ]
    
    for question in rag_questions:
        print(f"\n" + "="*80)
        rag.ask_question(question)
        print(f"\n" + "="*80)