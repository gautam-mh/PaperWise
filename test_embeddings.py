from src.pdf_extractor import PDFExtractor
from src.text_chunker import TextChunker
from src.embedding_manager import EmbeddingManager


def main():
    print("=== Complete Pipeline Test ===")
    
    # Step 1: Extract PDFs
    print("1. Extracting PDFs...")
    extractor = PDFExtractor()
    papers = extractor.extract_from_multiple_pdfs("data")
    
    if not papers:
        print("❌ No papers found! Add PDF files to the 'data' folder.")
        return
    
    print(f"✓ Extracted {len(papers)} papers")
    
    # Step 2: Chunk papers
    print("\n2. Chunking papers...")
    chunker = TextChunker(chunk_size=800, overlap=100)
    chunks = chunker.chunk_multiple_papers(papers)
    print(f"✓ Created {len(chunks)} chunks")
    
    # Step 3: Build embeddings
    print("\n3. Building embeddings and search index...")
    embedding_manager = EmbeddingManager()
    embedding_manager.build_index(chunks)
    
    # Step 4: Interactive search test
    print("\n4. Testing search functionality...")
    
    test_queries = [
    # Queries for space1.pdf (Planetary Boundaries paper)
    "What are the nine planetary boundaries identified in this research?",
    "How do climate change and biodiversity loss relate to planetary boundaries?",
    "What is the safe operating space for humanity according to planetary boundaries?",
    "Which planetary boundaries have already been transgressed by humanity?",
    "What are the control variables used for measuring planetary boundaries?",
    
    # Queries for space2.pdf (Medical imaging paper)
    "What is the difference between infratemporal fossa and masticator space?",
    "How do radiologists and surgeons differ in their terminology for head and neck spaces?",
    "What are the contents and boundaries of the parapharyngeal space?",
    "How does imaging help in staging head and neck cancers?",
    "What are the differential diagnoses for parapharyngeal space masses?",
    
    # Cross-paper and general queries
    "What methodologies are used in these research papers?",
    "What are the main findings and conclusions?",
    "How do these papers define boundaries and spaces?",
    "What are the clinical or practical implications discussed?",
    "What future research directions are suggested?"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Searching: '{query}'")
        results = embedding_manager.search(query, top_k=2)
        
        for i, result in enumerate(results, 1):
            chunk = result['chunk']
            score = result['similarity_score']
            
            print(f"  Result {i} (Score: {score:.3f})")
            print(f"    Paper: {chunk['paper_filename']}")
            print(f"    Section: {chunk['section']}")
            print(f"    Text: {chunk['text'][:150]}...")
            print()
    
    # Step 5: Save for later use
    print("5. Saving index for future use...")
    embedding_manager.save_index("saved_index")
    print("✓ Index saved to 'saved_index' folder")
    
    print("\n🎉 Pipeline test completed successfully!")

if __name__ == "__main__":
    main()