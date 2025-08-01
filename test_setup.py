# Test if all packages are installed correctly
try:
    import langchain
    import langchain_community
    import openai
    from langchain_community.document_loaders import PyMuPDFLoader
    from langchain_openai.embeddings import OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_community.llms import OpenAI
    from langchain.chains import RetrievalQA
    from dotenv import load_dotenv
    import os
    
    print("✅ All packages imported successfully!")
    
    # Test environment variables
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        print("✅ OpenAI API key found!")
    else:
        print("❌ OpenAI API key not found. Please add it to .env file")
        
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please install missing packages")