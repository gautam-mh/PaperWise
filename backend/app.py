from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
import json
from werkzeug.utils import secure_filename

# Add the parent directory to Python path to import from your project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your existing RAG system components
try:
    from rag_system import ResearchPaperRAG
except ImportError as e:
    print(f"Warning: Could not import ResearchPaperRAG: {e}")
    ResearchPaperRAG = None

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variable to store the RAG instance
rag_instance = None

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'Research Paper Reviewer API is running'
    })

@app.route('/api/upload', methods=['POST'])
def upload_paper():
    """Upload and process a research paper"""
    global rag_instance
    
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.lower().endswith('.pdf'):
            return jsonify({'error': 'Only PDF files are supported'}), 400
        
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Initialize the RAG system and process the documents
        if ResearchPaperRAG:
            if rag_instance is None:
                rag_instance = ResearchPaperRAG()
            
            # Load documents from the uploads folder
            documents = rag_instance.load_documents(app.config['UPLOAD_FOLDER'])
            
            # Chunk the documents
            chunks = rag_instance.chunk_documents()
            
            # Create embeddings and vectorstore
            rag_instance.create_embeddings_and_vectorstore(chunks)
            
            # Setup QA chain
            rag_instance.setup_qa_chain(model_name="llama2")
            
            return jsonify({
                'message': 'Paper uploaded and processed successfully',
                'filename': filename,
                'status': 'ready',
                'documents_loaded': len(documents),
                'chunks_created': len(chunks)
            })
        else:
            return jsonify({'error': 'ResearchPaperRAG not available'}), 500
            
    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/api/query', methods=['POST'])
def query_paper():
    """Query the uploaded paper"""
    global rag_instance
    
    try:
        if not rag_instance:
            return jsonify({'error': 'No paper uploaded yet'}), 400
        
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': 'Query text is required'}), 400
        
        query_text = data['query']
        
        # Get response from the RAG system
        response = rag_instance.ask_question(query_text)
        
        if response:
            return jsonify({
                'query': query_text,
                'response': response['result'],
                'sources': [
                    {
                        'filename': doc.metadata.get('filename', 'Unknown'),
                        'page': doc.metadata.get('page', 'Unknown'),
                        'content_preview': doc.page_content[:200] + '...'
                    }
                    for doc in response['source_documents']
                ],
                'status': 'success'
            })
        else:
            return jsonify({'error': 'Failed to get response from RAG system'}), 500
        
    except Exception as e:
        return jsonify({'error': f'Query failed: {str(e)}'}), 500

@app.route('/api/predefined-queries', methods=['GET'])
def get_predefined_queries():
    """Get list of predefined queries for testing"""
    queries = [
        "What is the main research question or hypothesis of this paper?",
        "What methodology was used in this research?",
        "What are the key findings or results?",
        "What are the limitations of this study?",
        "What future research directions are suggested?",
        "How does this work compare to existing literature?",
        "What is the significance or impact of this research?",
        "What datasets or tools were used in this study?"
    ]
    
    return jsonify({
        'queries': queries,
        'status': 'success'
    })

@app.route('/api/reset', methods=['POST'])
def reset_session():
    """Reset the current session"""
    global rag_instance
    
    try:
        rag_instance = None
        
        # Clean up uploaded files (optional)
        upload_folder = app.config['UPLOAD_FOLDER']
        for filename in os.listdir(upload_folder):
            file_path = os.path.join(upload_folder, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        
        return jsonify({
            'message': 'Session reset successfully',
            'status': 'reset'
        })
        
    except Exception as e:
        return jsonify({'error': f'Reset failed: {str(e)}'}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 50MB.'}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("Starting Research Paper Reviewer API Server...")
    print("Available endpoints:")
    print("  GET  /api/health - Health check")
    print("  POST /api/upload - Upload PDF paper")
    print("  POST /api/query - Query the paper")
    print("  GET  /api/predefined-queries - Get predefined queries")
    print("  POST /api/reset - Reset session")
    print("\nServer running on http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)