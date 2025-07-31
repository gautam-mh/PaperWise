from src.pdf_extractor import PDFExtractor
import os

def main():
    print("=== PDF Extraction Test ===")
    
    extractor = PDFExtractor()
    
    # Check if data folder exists
    if not os.path.exists("data"):
        print("Please add some PDF files to the 'data' folder first!")
        return
    
    # Extract from all PDFs in data folder
    papers = extractor.extract_from_multiple_pdfs("data")
    
    if papers:
        print(f"\nSuccessfully extracted text from {len(papers)} papers!")
        
        # Show summary of first paper
        first_paper = papers[0]
        print(f"\nSample from first paper:")
        print(f"Filename: {first_paper['filename']}")
        print(f"Title: {first_paper['title']}")
        print(f"Pages: {first_paper['num_pages']}")
        print(f"First 300 characters:\n{first_paper['full_text'][:300]}...")
    else:
        print("No papers were successfully processed.")

if __name__ == "__main__":
    main()