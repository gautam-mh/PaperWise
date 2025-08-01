import numpy as np
import faiss
import pickle
import os
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Optional

class EmbeddingManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding manager
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.model_name = model_name
        self.model = None
        self.index = None
        self.chunks_metadata = []
        self.embedding_dim = None
        
        # Load the model
        self._load_model()


    def _load_model(self):
        """Load the sentence transformer model"""
        print(f"Loading embedding model: {self.model_name}")
        try:
            self.model = SentenceTransformer(self.model_name)
            # Get embedding dimension by encoding a test sentence
            test_embedding = self.model.encode(["test"])
            self.embedding_dim = test_embedding.shape[1]
            print(f"✓ Model loaded successfully. Embedding dimension: {self.embedding_dim}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise


    def create_embeddings(self, chunks: List[Dict]) -> np.ndarray:
        """
        Create embeddings for a list of text chunks
        
        Args:
            chunks: List of chunk dictionaries containing text and metadata
            
        Returns:
            Numpy array of embeddings
        """
        print(f"Creating embeddings for {len(chunks)} chunks...")

        # Extract text from chunks
        texts = [chunk['text'] for chunk in chunks]
        
        # Create embeddings
        embeddings = self.model.encode(
            texts, 
            show_progress_bar=True,
            batch_size=32,
            convert_to_numpy=True
        )

        print(f"✓ Created embeddings with shape: {embeddings.shape}")
        return embeddings


    def build_index(self, chunks: List[Dict]) -> None:
        """
        Build FAISS index from chunks
        
        Args:
            chunks: List of chunk dictionaries
        """
        print("Building vector index...")
        
        # Create embeddings
        embeddings = self.create_embeddings(chunks)

        # Store chunks metadata
        self.chunks_metadata = chunks.copy()

        # Create FAISS Index
        self.index = faiss.IndexFlatIP(self.embedding_dim) # Inner Product (cosine similarity)

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)

        # Add embeddings to index
        self.index.add(embeddings.astype('float32'))

        print(f"✓ Index built with {self.index.ntotal} vectors")


    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search for similar chunks given a query
        
        Args:
            query: Search query string
            top_k: Number of top results to return
            
        Returns:
            List of dictionaries containing chunks and similarity scores
        """
        if self.index is None:
            raise ValueError("Index not built yet. Call build_index() first.")
        
        # Create embedding for query
        query_embedding = self.model.encode([query], convert_to_numpy=True)

        # Normalize query embedding
        faiss.normalize_L2(query_embedding)

        # Search in index
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)

        # Prepare results
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.chunks_metadata): # Valid index
                result = {
                    'chunk': self.chunks_metadata[idx].copy(),
                    'similarity_score': float(score),
                    'rank': i + 1
                }
                results.append(result)
        return results


    def save_index(self, save_dir: str) -> None:
        """
        Save the index and metadata to disk
        
        Args:
            save_dir: Directory to save files
        """
        os.makedirs(save_dir, exist_ok=True)

        if self.index is not None:
            # Save FAISS index
            index_path = os.path.join(save_dir, "faiss_index.bin")
            faiss.write_index(self.index, index_path)

            # Save metadata
            metadata_path = os.path.join(save_dir, "chunks_metadata.pkl")
            with open(metadata_path, 'wb') as f:
                pickle.dump(self.chunks_metadata, f)

            # Save config
            config_path = os.path.join(save_dir, "config.pkl")
            config = {
                'model_name': self.model_name,
                'embedding_dim': self.embedding_dim,
                'num_chunks': len(self.chunks_metadata)
            }
            with open(config_path, 'wb') as f:
                pickle.dump(config, f)
            
            print(f"✓ Index saved to {save_dir}")
        else:
            print("No index to save")
    

    def load_index(self, save_dir: str) -> None:
        """
        Load the index and metadata from disk
        
        Args:
            save_dir: Directory containing saved files
        """
        try:
            # Load config
            config_path = os.path.join(save_dir, "config.pkl")
            with open(config_path, 'rb') as f:
                config = pickle.load(f)
            
            # Verify model compatibility
            if config['model_name'] != self.model_name:
                print(f"Warning: Saved model ({config['model_name']}) differs from current ({self.model_name})")
            
            # Load FAISS index
            index_path = os.path.join(save_dir, "faiss_index.bin")
            self.index = faiss.read_index(index_path)
            
            # Load metadata
            metadata_path = os.path.join(save_dir, "chunks_metadata.pkl")
            with open(metadata_path, 'rb') as f:
                self.chunks_metadata = pickle.load(f)
            
            print(f"✓ Index loaded: {len(self.chunks_metadata)} chunks")
            
        except Exception as e:
            print(f"Error loading index: {e}")
            raise

    
    def get_stats(self) -> Dict:
        """Get statistics about the current index"""
        if self.index is None:
            return {"status": "No index built"}
        
        # Count chunks by paper
        papers = {}
        for chunk in self.chunks_metadata:
            paper = chunk['paper_filename']
            if paper not in papers:
                papers[paper] = 0
            papers[paper] += 1
        
        return {
            "total_chunks": len(self.chunks_metadata),
            "embedding_dimension": self.embedding_dim,
            "model_name": self.model_name,
            "papers": papers,
            "index_size": self.index.ntotal if self.index else 0
        }
    

# Test function
def test_embeddings():
    """Test the embedding system"""
    from pdf_extractor import PDFExtractor
    from text_chunker import TextChunker
    
    print("=== Embedding System Test ===")
    
    # Step 1: Extract and chunk papers
    print("Step 1: Extracting and chunking papers...")
    extractor = PDFExtractor()
    papers = extractor.extract_from_multiple_pdfs("data")
    
    if not papers:
        print("No papers found! Add PDFs to data folder.")
        return
    
    chunker = TextChunker()
    chunks = chunker.chunk_multiple_papers(papers)
    
    print(f"✓ Got {len(chunks)} chunks from {len(papers)} papers")
    
    # Step 2: Build embeddings and index
    print("\nStep 2: Building embeddings and index...")
    embedding_manager = EmbeddingManager()
    embedding_manager.build_index(chunks)
    
    # Step 3: Test search
    print("\nStep 3: Testing search...")
    test_queries = [
        "machine learning",
        "methodology",
        "results and findings",
        "abstract"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        results = embedding_manager.search(query, top_k=3)
        
        for result in results:
            chunk = result['chunk']
            score = result['similarity_score']
            print(f"  Score: {score:.3f} | Paper: {chunk['paper_filename']} | Section: {chunk['section']}")
            print(f"  Preview: {chunk['text'][:100]}...")
    
    # Step 4: Show statistics
    print(f"\n=== STATISTICS ===")
    stats = embedding_manager.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Step 5: Test save/load
    print(f"\nStep 5: Testing save/load...")
    embedding_manager.save_index("saved_index")
    
    # Create new manager and load
    new_manager = EmbeddingManager()
    new_manager.load_index("saved_index")
    
    # Test search with loaded index
    results = new_manager.search("machine learning", top_k=2)
    print(f"✓ Loaded index works: found {len(results)} results")

if __name__ == "__main__":
    test_embeddings()