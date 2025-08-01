# debug_extraction.py
from src.pdf_extractor import PDFExtractor
from src.text_chunker import TextChunker

def debug_extraction():
    print("=== Debugging PDF Extraction and Chunking ===")
    
    extractor = PDFExtractor()
    
    # Step 1: Test extract_from_multiple_pdfs method
    print("1. Testing extract_from_multiple_pdfs method...")
    try:
        papers_list = extractor.extract_from_multiple_pdfs("data")
        print(f"✓ Extracted {len(papers_list)} papers")
        
        for i, paper_data in enumerate(papers_list):
            print(f"\nPaper {i+1}:")
            print(f"  Filename: {paper_data['filename']}")
            print(f"  Full text length: {len(paper_data['full_text'])} chars")
            
            # Check content type
            if paper_data['filename'] == 'space2':
                text = paper_data['full_text'].lower()
                nlp_terms = ['neural network', 'word representations', 'vector', 'semantic', 'syntactic']
                nlp_count = sum(text.count(term) for term in nlp_terms)
                print(f"  NLP terms found: {nlp_count}")
                        
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Step 2: Test chunking
    print("\n2. Testing chunking...")
    try:
        papers_list = extractor.extract_from_multiple_pdfs("data")
        chunker = TextChunker()
        
        all_chunks = chunker.chunk_multiple_papers(papers_list)
        print(f"✓ Created {len(all_chunks)} total chunks")
        
        # Count chunks using the correct key: paper_filename
        space1_chunks = [c for c in all_chunks if c.get('paper_filename') == 'space1']
        space2_chunks = [c for c in all_chunks if c.get('paper_filename') == 'space2']
        
        print(f"  space1 chunks: {len(space1_chunks)}")
        print(f"  space2 chunks: {len(space2_chunks)}")
        
        # Check space2 chunks content
        if space2_chunks:
            print(f"\nFirst space2 chunk:")
            first_chunk = space2_chunks[0]
            print(f"  Paper title: {first_chunk.get('paper_title')}")
            print(f"  Section: {first_chunk.get('section')}")
            print(f"  Text preview: {first_chunk.get('text', '')[:150]}...")
            
            # Check if chunks contain NLP terms
            nlp_chunks = 0
            for chunk in space2_chunks:
                text = chunk.get('text', '').lower()
                if any(term in text for term in ['neural', 'vector', 'word', 'semantic', 'syntactic', 'language model']):
                    nlp_chunks += 1
            
            print(f"  Chunks with NLP terms: {nlp_chunks}/{len(space2_chunks)}")
            
            # Show a sample of space2 content
            print(f"\nSample space2 chunks:")
            for i, chunk in enumerate(space2_chunks[:3]):
                print(f"  Chunk {i+1} ({chunk.get('section')}): {chunk.get('text', '')[:100]}...")
        else:
            print("  No space2 chunks found!")
            
    except Exception as e:
        print(f"✗ Error in chunking: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_extraction()