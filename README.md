# Pharm-I: Local RAG-Based Chatbot for Pharmaceutical Robotic Systems

Pharm-I is a local Retrieval-Augmented Generation (RAG) chatbot developed as part of a Master's thesis in Artificial Intelligence and Automation Engineering.

The system is designed to provide question-answering support over technical documentation for pharmaceutical robotic systems while keeping document processing and language-model inference local.

## Overview

The application allows users to upload technical manuals in PDF format and ask natural-language questions about their contents.

The RAG pipeline retrieves relevant document sections and provides them as context to a locally hosted Large Language Model (LLM), which generates an answer based only on the retrieved documentation.

## Architecture

The main workflow is:

PDF Manuals  
↓  
Document Loading  
↓  
Text Chunking  
↓  
Embedding Generation  
↓  
ChromaDB Vector Store  
↓  
MMR Retrieval  
↓  
Context Construction  
↓  
Local LLM via Ollama  
↓  
Answer + Document References

## Technologies

- Python
- Streamlit
- LangChain
- ChromaDB
- Hugging Face / Sentence Transformers
- Ollama
- SQLite
- NumPy
- NLTK BLEU
- ROUGE Score

## RAG Configuration

The main application uses:

- LLM: `gpt-oss:20b`
- Embedding model: `intfloat/multilingual-e5-small`
- Retrieval: Maximal Marginal Relevance (MMR)
- Top-K retrieved chunks: 5
- Chunk size: 900 characters
- Chunk overlap: 120 characters
- ChromaDB vector storage
- Local Ollama inference

## Repository Files

### `app.py`

Main Streamlit application implementing the local RAG chatbot.

Features include:

- PDF manual upload
- Document chunking
- Vector embedding generation
- ChromaDB indexing
- MMR-based retrieval
- Local LLM inference using Ollama
- Source and page references
- Local chat history using SQLite

### `pdf2sql.py`

Utility for extracting structured Question/Answer pairs from PDF documents and storing them in an SQLite database.

This script was used to support construction of the evaluation dataset.

### `listemb.py`

Experimental pipeline for evaluating different embedding models within the RAG architecture.

Evaluation includes:

- Semantic similarity
- BLEU
- ROUGE-1
- ROUGE-2
- ROUGE-L
- Response latency

### `modeldb.py`

Experimental framework for evaluating different local Large Language Models and embedding configurations.

Generated answers and evaluation metrics are stored in SQLite for later analysis.

## Running the Application

### 1. Install Ollama

Install Ollama and make sure the Ollama server is running.

### 2. Download the required model

```bash
ollama pull gpt-oss:20b
```

### 3. Install Python dependencies

```bash
pip install streamlit langchain langchain-community langchain-chroma langchain-ollama chromadb sentence-transformers pypdf numpy nltk rouge-score
```

### 4. Start the application

```bash
streamlit run app.py
```

### 5. Upload Manuals

Upload one or more PDF manuals through the Streamlit interface.

The documents are processed locally and indexed using ChromaDB.

### 6. Ask Questions

Enter a technical question in the application.

The system retrieves relevant sections from the uploaded manuals and generates an answer using the local LLM.

## Evaluation

The experimental framework was developed to compare different embedding models and locally hosted LLMs.

The evaluation considered:

- Semantic similarity
- BLEU
- ROUGE
- Response latency

The experiments were performed using a manually curated technical Question/Answer dataset.

## Privacy and Local Deployment

The system is designed for local deployment.

Document processing, vector retrieval, database operations, and LLM inference can be performed locally without requiring technical manuals to be sent to external cloud-based language-model APIs.

## Data Availability

The technical manuals, proprietary Question/Answer dataset, experimental databases, and other confidential industrial documentation used during the research are not included in this repository.

The repository contains only the source code required to demonstrate the architecture and experimental methodology.

## Academic Project

This repository accompanies the Master's thesis:

**Design and Evaluation of a Local RAG-Based Chatbot for Pharmaceutical Robotic Systems**

Master's Degree in Artificial Intelligence and Automation Engineering  
University of Siena

## Author

**Syed Yasir Ahmed**

## Disclaimer

This project is intended for research and technical information-assistance purposes. It is not designed to directly control pharmaceutical machinery or replace official operating procedures, safety documentation, or qualified technical personnel.
