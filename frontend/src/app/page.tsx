'use client';

import { useState } from 'react';
import { Upload, FileText, MessageSquare, Trash2, Search } from 'lucide-react';
import axios from 'axios';

interface UploadedFile {
  name: string;
  size: number;
  status: 'uploading' | 'ready' | 'error';
}

interface QueryResult {
  query: string;
  response: string;
  timestamp: string;
}

export default function Home() {
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [query, setQuery] = useState('');
  const [isQuerying, setIsQuerying] = useState(false);
  const [queryResults, setQueryResults] = useState<QueryResult[]>([]);
  const [predefinedQueries, setPredefinedQueries] = useState<string[]>([]);

  const API_BASE = 'http://localhost:5000/api';

  // Load predefined queries on component mount
  const loadPredefinedQueries = async () => {
    try {
      const response = await axios.get(`${API_BASE}/predefined-queries`);
      setPredefinedQueries(response.data.queries);
    } catch (error) {
      console.error('Failed to load predefined queries:', error);
    }
  };

  // Handle file upload (batch)
  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (!files || files.length === 0) return;

    setIsUploading(true);
    const newFiles: UploadedFile[] = [];

    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      newFiles.push({
        name: file.name,
        size: file.size,
        status: 'uploading'
      });
    }

    setUploadedFiles(prev => [...prev, ...newFiles]);

    // Upload files one by one
    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      const formData = new FormData();
      formData.append('file', file);

      try {
        await axios.post(`${API_BASE}/upload`, formData, {
          headers: { 'Content-Type': 'multipart/form-data' }
        });

        // Update file status to ready
        setUploadedFiles(prev => 
          prev.map(f => 
            f.name === file.name ? { ...f, status: 'ready' } : f
          )
        );
      } catch (error) {
        // Update file status to error
        setUploadedFiles(prev => 
          prev.map(f => 
            f.name === file.name ? { ...f, status: 'error' } : f
          )
        );
        console.error(`Failed to upload ${file.name}:`, error);
      }
    }

    setIsUploading(false);
    loadPredefinedQueries(); // Load queries after successful upload
  };

  // Handle query submission
  const handleQuery = async (queryText: string) => {
    if (!queryText.trim() || uploadedFiles.length === 0) return;

    setIsQuerying(true);
    try {
      const response = await axios.post(`${API_BASE}/query`, {
        query: queryText
      });

      const newResult: QueryResult = {
        query: queryText,
        response: response.data.response,
        timestamp: new Date().toLocaleTimeString()
      };

      setQueryResults(prev => [newResult, ...prev]);
      setQuery(''); // Clear input after successful query
    } catch (error) {
      console.error('Query failed:', error);
      alert('Query failed. Please try again.');
    }
    setIsQuerying(false);
  };

  // Reset session
  const handleReset = async () => {
    try {
      await axios.post(`${API_BASE}/reset`);
      setUploadedFiles([]);
      setQueryResults([]);
      setPredefinedQueries([]);
    } catch (error) {
      console.error('Reset failed:', error);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-6xl mx-auto px-4">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            Research Paper Reviewer
          </h1>
          <p className="text-gray-600">
            Upload research papers and ask questions using AI-powered analysis
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Left Column - Upload & Files */}
          <div className="space-y-6">
            {/* File Upload */}
            <div className="bg-white rounded-lg shadow-md p-6">
              <h2 className="text-xl font-semibold mb-4 flex items-center">
                <Upload className="mr-2" size={20} />
                Upload Research Papers
              </h2>
              
              <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center">
                <input
                  type="file"
                  multiple
                  accept=".pdf"
                  onChange={handleFileUpload}
                  className="hidden"
                  id="file-upload"
                  disabled={isUploading}
                />
                <label
                  htmlFor="file-upload"
                  className={`cursor-pointer ${isUploading ? 'opacity-50' : ''}`}
                >
                  <FileText className="mx-auto mb-4 text-gray-400" size={48} />
                  <p className="text-lg font-medium text-gray-700">
                    {isUploading ? 'Uploading...' : 'Click to upload PDF files'}
                  </p>
                  <p className="text-sm text-gray-500 mt-2">
                    Select multiple PDF files (up to 50MB each)
                  </p>
                </label>
              </div>
            </div>

            {/* Uploaded Files List */}
            {uploadedFiles.length > 0 && (
              <div className="bg-white rounded-lg shadow-md p-6">
                <div className="flex justify-between items-center mb-4">
                  <h3 className="text-lg font-semibold">Uploaded Files</h3>
                  <button
                    onClick={handleReset}
                    className="text-red-600 hover:text-red-800 flex items-center"
                  >
                    <Trash2 size={16} className="mr-1" />
                    Reset
                  </button>
                </div>
                
                <div className="space-y-2">
                  {uploadedFiles.map((file, index) => (
                    <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded">
                      <div>
                        <p className="font-medium">{file.name}</p>
                        <p className="text-sm text-gray-500">
                          {(file.size / 1024 / 1024).toFixed(2)} MB
                        </p>
                      </div>
                      <div className={`px-2 py-1 rounded text-xs font-medium ${
                        file.status === 'ready' ? 'bg-green-100 text-green-800' :
                        file.status === 'uploading' ? 'bg-yellow-100 text-yellow-800' :
                        'bg-red-100 text-red-800'
                      }`}>
                        {file.status}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Right Column - Query & Results */}
          <div className="space-y-6">
            {/* Query Interface */}
            <div className="bg-white rounded-lg shadow-md p-6">
              <h2 className="text-xl font-semibold mb-4 flex items-center">
                <MessageSquare className="mr-2" size={20} />
                Ask Questions
              </h2>

              <div className="space-y-4">
                <div className="flex space-x-2">
                  <input
                    type="text"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder="Ask a question about your papers..."
                    className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    onKeyPress={(e) => e.key === 'Enter' && handleQuery(query)}
                    disabled={isQuerying || uploadedFiles.length === 0}
                  />
                  <button
                    onClick={() => handleQuery(query)}
                    disabled={isQuerying || !query.trim() || uploadedFiles.length === 0}
                    className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center"
                  >
                    <Search size={16} className="mr-1" />
                    {isQuerying ? 'Asking...' : 'Ask'}
                  </button>
                </div>

                {/* Predefined Queries */}
                {predefinedQueries.length > 0 && (
                  <div>
                    <p className="text-sm font-medium text-gray-700 mb-2">Quick Questions:</p>
                    <div className="flex flex-wrap gap-2">
                      {predefinedQueries.slice(0, 4).map((predefinedQuery, index) => (
                        <button
                          key={index}
                          onClick={() => handleQuery(predefinedQuery)}
                          disabled={isQuerying || uploadedFiles.length === 0}
                          className="px-3 py-1 text-sm bg-gray-100 text-gray-700 rounded-full hover:bg-gray-200 disabled:opacity-50"
                        >
                          {predefinedQuery.length > 50 ? 
                            predefinedQuery.substring(0, 50) + '...' : 
                            predefinedQuery
                          }
                        </button>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Query Results */}
            {queryResults.length > 0 && (
              <div className="bg-white rounded-lg shadow-md p-6">
                <h3 className="text-lg font-semibold mb-4">Results</h3>
                <div className="space-y-4 max-h-96 overflow-y-auto">
                  {queryResults.map((result, index) => (
                    <div key={index} className="border-l-4 border-blue-500 pl-4 py-2">
                      <p className="font-medium text-gray-900 mb-1">
                        Q: {result.query}
                      </p>
                      <p className="text-gray-700 mb-2">
                        A: {result.response}
                      </p>
                      <p className="text-xs text-gray-500">
                        {result.timestamp}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}