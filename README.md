# OCR_RAG
This script combines several powerful tools for handwritten text recognition and information retrieval, primarily using Google Cloud Vision for OCR (Optical Character Recognition), Pinecone for similarity search, and OpenAI's GPT models for natural language processing.
Google Cloud Vision: Used for extracting text from images. The API credentials are loaded, and the client is initialized.
Pinecone: A vector database is initialized with the specified API key and environment. An index is created for storing and querying vector embeddings.
OpenAI: The API key is set for making calls to GPT models for text generation and embeddings.
The script is designed to be run in a Streamlit application, providing a user-friendly interface for uploading images and displaying results.
