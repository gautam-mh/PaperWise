from src.pdf_extractor import PDFExtractor
from src.text_chunker import TextChunker
from src.embedding_manager import EmbeddingManager
import os


def test_complete_pipeline():
    print("=== Complete Pipeline Test ===")
    
    # Step 1: Extract PDFs from data folder
    print("1. Extracting PDFs...")
    extractor = PDFExtractor()
    papers_list = extractor.extract_from_multiple_pdfs("data")
    
    pdf_files = [f for f in os.listdir("data") if f.endswith('.pdf')]
    print(f"Found {len(pdf_files)} PDF files")
    
    for paper_data in papers_list:
        print(f"Processing: {paper_data['filename']}.pdf")
        print(f"✓ Successfully extracted text from {paper_data['filename']}.pdf")
    
    print(f"✓ Extracted {len(papers_list)} papers")
    
    # Step 2: Chunk papers
    print("\n2. Chunking papers...")
    chunker = TextChunker()
    all_chunks = chunker.chunk_multiple_papers(papers_list)
    
    # Count chunks by paper using the correct key
    space1_chunks = [c for c in all_chunks if c.get('paper_filename') == 'space1']
    space2_chunks = [c for c in all_chunks if c.get('paper_filename') == 'space2']
    
    print(f"Created {len(space1_chunks)} chunks for space1")
    print(f"Created {len(space2_chunks)} chunks for space2")
    print(f"Total chunk created: {len(all_chunks)}")
    print(f"✓ Created {len(all_chunks)} chunks")
    
    # Step 3: Build embeddings
    print("\n3. Building embeddings and search index...")
    embedding_manager = EmbeddingManager()
    embedding_manager.build_index(all_chunks)
    
    # Step 4: Test searches with appropriate queries for each paper
    print("\n4. Testing search functionality...")
    
    # Queries for space1.pdf (planetary boundaries paper)
    planetary_queries = [
        "What are the nine planetary boundaries identified in this research?",
        "How do climate change and biodiversity loss relate to planetary boundaries?",
        "What is the safe operating space for humanity according to planetary boundaries?",
        "Which planetary boundaries have already been transgressed by humanity?",
        "What are the control variables used for measuring planetary boundaries?"
    ]
    
    # Queries for space2.pdf (NLP word representations paper)  
    nlp_queries = [
        "What are word representations and how are they learned?",
        "How do neural network language models create word vectors?",
        "What is the vector offset method for solving analogy questions?",
        "How do word vectors capture syntactic and semantic regularities?",
        "What is the relationship between King, Man, Woman and Queen in vector space?",
        "How does the recurrent neural network language model work?",
        "What are the results on syntactic analogy questions?",
        "How does this method perform on semantic similarity tasks?"
    ]
    
    # General queries that might work for both
    general_queries = [
        "What methodologies are used in these research papers?",
        "What are the main findings and conclusions?",
        "What datasets were used for evaluation?",
        "What are the experimental results reported?",
        "What future research directions are suggested?"
    ]
    
    all_test_queries = planetary_queries + nlp_queries + general_queries
    
    for query in all_test_queries:
        print(f"\n🔍 Searching: '{query}'")
        results = embedding_manager.search(query, top_k=2)
        
        for i, result in enumerate(results, 1):
            chunk = result['chunk']
            score = result['similarity_score']
            print(f"  Result {i} (Score: {score:.3f})")
            print(f"    Paper: {chunk.get('paper_filename', 'unknown')}")
            print(f"    Section: {chunk.get('section', 'unknown')}")
            print(f"    Text: {chunk.get('text', '')[:100]}...")
    
    # Step 5: Save index
    print("\n5. Saving index for future use...")
    embedding_manager.save_index("saved_index")
    print("✓ Index saved to 'saved_index' folder")
    
    print("\n🎉 Pipeline test completed successfully!")

if __name__ == "__main__":
    test_complete_pipeline()