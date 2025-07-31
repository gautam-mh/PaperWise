from src.pdf_extractor import PDFExtractor
from src.text_chunker import TextChunker

def main():
    print("=== Text Chunking Test ===")
    
    # Step 1: Extract text from PDFs
    print("Step 1: Extracting text from PDFs...")
    extractor = PDFExtractor()
    papers = extractor.extract_from_multiple_pdfs("data")
    
    if not papers:
        print("No papers found! Please add PDF files to the 'data' folder.")
        return
    
    print(f"✓ Extracted text from {len(papers)} papers")

    # Step 2: Chunk the papers
    print("\nStep 2: Chunking papers...")
    chunker = TextChunker(chunk_size=800, overlap=100)
    all_chunks = chunker.chunk_multiple_papers(papers)
    
    # Step 3: Show results
    print(f"\n=== RESULTS ===")
    print(f"Papers processed: {len(papers)}")
    print(f"Total chunks: {len(all_chunks)}")
    
    # Show breakdown by paper
    for paper in papers:
        paper_chunks = [c for c in all_chunks if c['paper_filename'] == paper['filename']]
        print(f"  {paper['filename']}: {len(paper_chunks)} chunks")
    
    # Show sample chunk
    if all_chunks:
        print(f"\n=== SAMPLE CHUNK ===")
        sample = all_chunks[0]
        print(f"From: {sample['paper_filename']}")
        print(f"Section: {sample['section']}")
        print(f"Length: {sample['char_count']} chars")
        print(f"Preview:\n{sample['text'][:300]}...")

if __name__ == "__main__":
    main()