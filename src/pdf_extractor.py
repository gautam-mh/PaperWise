import os
from typing import Dict, List

# Alternative import if the above doesn't work:
import pymupdf

class PDFExtractor:
    def __init__(self):
        pass

    def extract_text_from_pdf(self, pdf_path: str) -> Dict[str, str]:
        """
        Extract text from a PDF file and return it with metadata
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing extracted text and metadata
        """
        try:
            # Open the PDF
            doc = pymupdf.open(pdf_path)
            
            # Extract basic metadata
            metadata = doc.metadata
            
            # Extract text from all pages
            full_text = ""
            page_texts = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_text = page.get_text()
                page_texts.append(page_text)
                full_text += page_text + "\n"
            
            doc.close()
            
            # Get filename without extension
            filename = os.path.basename(pdf_path).replace('.pdf', '')
            
            return {
                'filename': filename,
                'full_text': full_text,
                'page_texts': page_texts,
                'num_pages': len(page_texts),
                'title': metadata.get('title', filename),
                'author': metadata.get('author', 'Unknown'),
                'subject': metadata.get('subject', ''),
                'pdf_path': pdf_path
            }
            
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {str(e)}")
            return None

    def extract_from_multiple_pdfs(self, pdf_folder:str) -> List[Dict[str, str]]:
        """
        Extract text from multiple PDF files in a folder and return a list of dictionaries containing extracted text and metadata
        
        Args:
            pdf_folder: Path to the folder containing PDF files
            
        Returns:
            List of dictionaries containing extracted text and metadata
        """
        extracted_papers = []

        # Get all PDF files in the folder
        pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith('.pdf')]
        
        print(f"Found {len(pdf_files)} PDF files")
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(pdf_folder, pdf_file)
            print(f"Processing: {pdf_file}")

            extracted_data = self.extract_text_from_pdf(pdf_path)
            if extracted_data:
                extracted_papers.append(extracted_data)
                print(f"✓ Successfully extracted text from {pdf_file}")
            else:
                print(f"✗ Failed to extract text from {pdf_file}")
        
        return extracted_papers

    # Test function
    def test_extractor():
        """Test the PDF extractor with a sample file"""
        extractor = PDFExtractor()

        # Test with data folder
        data_folder = "data"
        if os.path.exists(data_folder):
            papers = extractor.extract_from_multiple_pdfs(data_folder)
            
            if papers:
                print(f"\n=== EXTRACTION RESULTS ===")
                for paper in papers:
                    print(f"Title: {paper['title']}")
                    print(f"Author: {paper['author']}")
                    print(f"Pages: {paper['num_pages']}")
                    print(f"Text length: {len(paper['full_text'])} characters")
                    print(f"First 200 characters: {paper['full_text'][:200]}...")
                    print("-" * 50)
            else:
                print("No PDFs found or extraction failed")
        else:
            print("Data folder not found")

    if __name__ == "__main__":
        test_extractor()