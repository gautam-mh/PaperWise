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
        self.chunks = []
        
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

        self.chunks = chunks
        
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

    
    def create_embeddings_and_vectorstore(self, chunks = None):
        """
        Create embeddings for chunks and build vector store
        """

        try:
            # Use provided chunks or stored chunks
            if chunks is None:
                if hasattr(self, 'chunks') and self.chunks:
                    chunks = self.chunks
                else:
                    print("❌ No chunks available. Run chunk_documents() first.")
                    return None
            
            print(f"🧠 Creating embeddings for {len(chunks)} chunks...")
            
            # Initialize HuggingFace embeddings (free, local)
            print("🔧 Initializing HuggingFace embeddings...")
            self.embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            print("✅ HuggingFace embeddings initialized")
            
            # Create vector store from documents
            print("🔍 Building FAISS vector store...")
            self.vectorstore = FAISS.from_documents(
                documents=chunks,
                embedding=self.embeddings
            )
            
            print("✅ Vector store created successfully!")
            return self.vectorstore
        
        except Exception as e:
            print(f"❌ ERROR in create_embeddings_and_vectorstore: {str(e)}")
            print(f"❌ ERROR type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            raise e
    
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
        try:
            if self.vectorstore is None:
                print("❌ No vector store available. Create embeddings first.")
                return
            
            print(f"🤖 Setting up QA chain with Ollama model: {model_name}")
            
            # Initialize Ollama LLM
            print("🔧 Initializing Ollama LLM...")
            llm = OllamaLLM(
                model=model_name,
                temperature=0.1,
            )
            print("✅ Ollama LLM initialized")
            
            # Create custom prompt template for RAG
            prompt_template = """
            You are a helpful research assistant. Use the following pieces of context from research papers to answer the question at the end. Answer precisely and do not beat around the bush. Answer pointwise wherever possible. Do not answer in a big paragraph ofmore than 5 lines. If it exceeds more than 5 lines, try to answer in next paragraph 

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
            print("✅ Prompt template created")
            
            # Create the QA chain
            print("🔗 Creating QA chain...")
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=self.vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 4}
                ),
                chain_type_kwargs={"prompt": PROMPT},
                return_source_documents=True
            )
            
            print("✅ QA chain setup complete!")
            return self.qa_chain
            
        except Exception as e:
            print(f"❌ ERROR in setup_qa_chain: {str(e)}")
            print(f"❌ ERROR type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            raise e

    
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


    def save_chunk_analysis(self, output_file="chunk_analysis_detailed.txt"):
        """Save detailed chunk analysis with full content to a file"""
        if not hasattr(self, 'chunks') or not self.chunks:
            with open(output_file, 'w') as f:
                f.write("❌ No chunks available. Run chunk_documents() first.\n")
            return
        
        chunks = self.chunks
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("📊 DETAILED CHUNK ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")
            
            # Basic stats
            lengths = [len(chunk.page_content) for chunk in chunks]
            words = [len(chunk.page_content.split()) for chunk in chunks]
            
            f.write("📈 QUICK STATISTICS:\n")
            f.write(f"  Total chunks: {len(chunks)}\n")
            f.write(f"  Avg length: {sum(lengths)/len(lengths):.0f} characters\n")
            f.write(f"  Avg words: {sum(words)/len(words):.0f} words\n")
            f.write(f"  Min/Max length: {min(lengths)} / {max(lengths)}\n\n")
            
            # By source summary
            by_source = {}
            for chunk in chunks:
                source = chunk.metadata.get('source', 'unknown').split('/')[-1]
                by_source[source] = by_source.get(source, 0) + 1
            
            f.write("📁 CHUNKS BY SOURCE:\n")
            for source, count in by_source.items():
                f.write(f"  {source}: {count} chunks\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("📄 ALL CHUNKS WITH FULL CONTENT\n")
            f.write("="*80 + "\n\n")
            
            # Show all chunks with full content
            current_source = None
            for i, chunk in enumerate(chunks):
                source = chunk.metadata.get('source', 'unknown').split('/')[-1]
                page = chunk.metadata.get('page', 'N/A')
                
                # Add source separator when source changes
                if source != current_source:
                    if current_source is not None:
                        f.write("\n" + "🔄 " + "="*70 + " 🔄\n\n")
                    f.write(f"📖 SOURCE: {source}\n")
                    f.write("─" * 80 + "\n\n")
                    current_source = source
                
                # Chunk header
                f.write(f"┌─ CHUNK #{i+1:03d} ─────────────────────────────────────────────────────┐\n")
                f.write(f"│ Source: {source:<30} │ Page: {str(page):<10} │\n")
                f.write(f"│ Length: {len(chunk.page_content):<6} chars │ Words: {len(chunk.page_content.split()):<6} │ Metadata: {str(chunk.metadata):<20} │\n")
                f.write(f"└─────────────────────────────────────────────────────────────────────┘\n\n")
                
                # Full chunk content
                f.write("CONTENT:\n")
                f.write("─" * 40 + "\n")
                f.write(chunk.page_content)
                f.write("\n" + "─" * 40 + "\n\n")
                
                # Add visual separator between chunks
                f.write("▼" * 80 + "\n\n")
            
            f.write(f"\n" + "="*80 + "\n")
            f.write(f"📊 REPORT SUMMARY:\n")
            f.write(f"  Total chunks analyzed: {len(chunks)}\n")
            f.write(f"  Sources processed: {len(by_source)}\n")
            f.write(f"  Report generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n")
        
        print(f"✅ Detailed chunk analysis saved to: {output_file}")
        print(f"📄 File contains {len(chunks)} chunks with full content for manual inspection")

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