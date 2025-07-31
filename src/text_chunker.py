import re
from typing import List, Dict, Tuple

class TextChunker:
    def __init__(self, chunk_size: int = 800, overlap: int = 100):
        """
        Initialize the text chunker
        
        Args:
            chunk_size: Maximum number of characters per chunk
            overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

        
        """
        different types of section patterns:
            "ABSTRACT" (all caps)
            "1. Introduction" (numbered)
            "II. Methods" (roman numerals)
            "abstract" (lowercase)
        """
        # More precise section patterns
        self.section_patterns = [
            # Direct section names
            r'^\s*(?:abstract|introduction|related work|literature review|methodology|methods|approach|implementation|experiments?|results?|evaluation|discussion|conclusions?|references|bibliography|acknowledgments?|appendix)\s*\.?\s*$',
            # Numbered sections
            r'^\s*\d+\.?\s+(?:abstract|introduction|related work|literature review|methodology|methods|approach|implementation|experiments?|results?|evaluation|discussion|conclusions?|references|bibliography|acknowledgments?|appendix)\s*\.?\s*$',
            # Roman numeral sections
            r'^\s*(?:i{1,3}|iv|v|vi{0,3}|ix|x)\.?\s+(?:abstract|introduction|related work|literature review|methodology|methods|approach|implementation|experiments?|results?|evaluation|discussion|conclusions?|references|bibliography|acknowledgments?|appendix)\s*\.?\s*$',
            # All caps sections
            r'^\s*(?:ABSTRACT|INTRODUCTION|RELATED WORK|LITERATURE REVIEW|METHODOLOGY|METHODS|APPROACH|IMPLEMENTATION|EXPERIMENTS?|RESULTS?|EVALUATION|DISCUSSION|CONCLUSIONS?|REFERENCES|BIBLIOGRAPHY|ACKNOWLEDGMENTS?|APPENDIX)\s*\.?\s*$'
        ]
    
    def detect_sections(self, text: str) -> List[Dict[str, any]]:
        """
        Try to detect sections in the research paper
        
        Args:
            text: Full text of the paper
            
        Returns:
            List of sections with their positions and titles
        """
        sections = []
        lines = text.split('\n')

        for i, line in enumerate(lines):
            line_clean = line.strip().lower()
            
            # Skip very short lines
            if len(line_clean) < 3 or len(line_clean) > 100:
                continue

            # Check if line matches any section pattern
            for pattern in self.section_patterns:
                match = re.search(pattern, line_clean, re.IGNORECASE)
                if match:
                    # Extract just the section name, not the whole line
                    section_name = self.extract_section_name(line_clean)

                    sections.append({
                        'title': section_name,
                        'line_number': i,
                        'start_char': sum(len(lines[j]) + 1 for j in range(i))
                    })
                    break
        
        return sections

    def extract_section_name(self, line: str) -> str:
        """
        Extract clean section name from a line
        
        Args:
            line: Line containing section header
            
        Returns:
            Clean section name
        """
        line_clean = line.strip()
        
        # Common section names to look for
        section_names = [
            'abstract', 'introduction', 'related work', 'literature review',
            'methodology', 'methods', 'approach', 'implementation', 
            'experiments', 'experiment', 'results', 'evaluation', 
            'discussion', 'conclusion', 'conclusions', 'references', 
            'bibliography', 'acknowledgments', 'appendix'
        ]
        
        # Remove common prefixes (numbers, roman numerals, etc.)
        cleaned = re.sub(r'^[\d\.\s\-\)\(ivxlc]+', '', line_clean, flags=re.IGNORECASE)
        cleaned = cleaned.strip()
        
        # Look for section names in the cleaned line
        for section_name in section_names:
            if section_name.lower() in cleaned.lower()[:20]:  # Check first 20 chars
                return section_name.upper()
        
        # If no standard section found, try to extract first few words
        words = cleaned.split()
        if words:
            # Take first 1-2 words that look like a section header
            if len(words[0]) > 2:
                return words[0].upper()
        
        return "UNKNOWN SECTION" 
    
    def chunk_by_sections(self, text: str, paper_info: Dict[str, any]) -> List[Dict[str, any]]:
        """
        Chunk text by detected sections first, then by size if needed

        Args:
            text: Full text to chunk
            paper_info: Information about the paper

        Returns:
            List of chunks with metadata
        """
        chunks = []
        sections = self.detect_sections(text)

        if not sections:
            # No sections detected, use simple chunking
            return self.chunk_by_size(text, paper_info)

        #Add end positions to sections
        for i in range(len(sections)):
            if i < len(sections) - 1:
                sections[i]['end_char'] = sections[i + 1]['start_char']
            else:
                sections[i]['end_char'] = len(text)

        # Extract text for each section
        for i, section in enumerate(sections):
            section_text = text[section['start_char']:section['end_char']].strip()
            
            # If section is too long, chunk it further
            if len(section_text) > self.chunk_size:
                section_chunks = self.chunk_by_size(section_text, paper_info, section['title'])
                chunks.extend(section_chunks)
            else:
                # Section fits in one chunk
                chunk = {
                    'text': section_text,
                    'chunk_id': f"{paper_info['filename']}_section_{i}",
                    'paper_title': paper_info['title'],
                    'paper_filename': paper_info['filename'],
                    'section': section['title'],
                    'chunk_type': 'section',
                    'char_count': len(section_text),
                    'chunk_index': i
                }
                chunks.append(chunk)
        
        return chunks

    def chunk_by_size(self, text:str, paper_info: Dict, section_title: str = None) -> List[Dict[str, any]]:
        """
        Chunk text by size with overlap

        Args:
            text: Text to chunk
            paper_info: Inforation about the paper
            section_title: Title of the section (if applicable)

        Returns:
            List of chunks with metadata
        """
        chunks = []
        start = 0
        chunk_index = 0

        while start < len(text):
            # Calculate end position
            end = start + self.chunk_size

            # If this isn't the last chunk, try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings near the chunk boundary
                sentence_end = text.rfind('.', start, end)
                if sentence_end > start + self.chunk_size // 2:
                    end = sentence_end + 1
            
            chunk_text = text[start:end].strip()

            if chunk_text: # Only add non-empty chunks
                chunk = {
                    'text': chunk_text,
                    'chunk_id': f"{paper_info['filename']}_chunk_{chunk_index}",
                    'paper_title': paper_info['title'],
                    'paper_filename': paper_info['filename'],
                    'section': section_title or 'Unknown',
                    'chunk_type': 'size_based',
                    'char_count': len(chunk_text),
                    'chunk_index': chunk_index
                }
                chunks.append(chunk)
                chunk_index += 1
            
            # Move start position (with overlap)
            start = end - self.overlap

            # Prevent infinite loop
            if start >= len(text):
                break
        
        return chunks
        
    def chunk_paper(self, paper_data: Dict) -> List[Dict[str,any]]:
        """
        Main method to chunk a paper

        Args:
            papaer_data: Dictionary containing paper text and metadata
        
        Returns:
            List of chunks
        """
        text = paper_data['full_text']

        # Try section-based chunking first
        chunks = self.chunk_by_sections(text, paper_data)
        
        print(f"Created {len(chunks)} chunks for {paper_data['filename']}")
        
        return chunks

    def chunk_multiple_papers(self, papers_data: List[Dict]) -> List[Dict[str, any]]:
        """
        Chunk multiple papers

        Args:
            papers_data: List of paper data dictionaries
        
        Returns:
            List of all chunks from all papers
        """
        all_chunks=[]

        for paper_data in papers_data:
            paper_chunks = self.chunk_paper(paper_data)
            all_chunks.extend(paper_chunks)
        
        print(f"Total chunk created: {len(all_chunks)}")
        return all_chunks


# Test function
def test_chunker():
    """Test the chunker with extracted papers"""
    from pdf_extractor import PDFExtractor

    # First extract text from PDFS
    extractor = PDFExtractor()
    papers = extractor.extract_from_multiple_pdfs("data")

    if not papers:
        print("No papers found. Please add PDFs to the data folder.")
        return
    
    # Initialize chunker
    chunker = TextChunker(chunk_size=800, overlap=100)

    # Chunk all papers
    all_chunks = chunker.chunk_multiple_papers(papers)

    # Show results
    print(f"\n=== CHUNKING RESULTS ===")
    print(f"Total papers processed: {len(papers)}")
    print(f"Total chunks created: {len(all_chunks)}")

    # Show sample chunks
    print(f"\n=== SAMPLE CHUNKS ===")
    for i, chunk in enumerate(all_chunks[:3]):  # Show first 3 chunks
        print(f"\nChunk {i+1}:")
        print(f"Paper: {chunk['paper_filename']}")
        print(f"Section: {chunk['section']}")
        print(f"Type: {chunk['chunk_type']}")
        print(f"Length: {chunk['char_count']} characters")
        print(f"Text preview: {chunk['text'][:200]}...")
        print("-" * 50)

if __name__ == "__main__":
    test_chunker()