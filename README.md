# Dr. X NLP Pipeline

## Overview

This project implements a **Retrieval-Augmented Generation (RAG) system** designed to process and analyze Dr. X's publications across multiple formats (PDF, DOCX, CSV, Excel). The goal is to extract meaningful information and provide relevant answers through a question-answering system powered by **LLaMA** and **FAISS**.

## Key Features

- **Text Extraction**: Supports multiple file formats (`.pdf`, `.docx`, `.csv`, `.xlsx`, `.xlsm`).
- **Chunking & Tokenization**: Efficiently breaks down large documents into smaller chunks using **Tiktoken** to handle long documents.
- **Embedding**: Embeds text chunks using **SentenceTransformer** for efficient similarity-based retrieval.
- **Retrieval-Augmented Generation (RAG)**: Uses **FAISS** for fast vector-based search, combined with **LLaMA** for contextual question answering.
- **Translation & Summarization**: Provides translation and concise summaries for multilingual research data.
- **Error Handling**: The system gracefully handles potential errors in document extraction and processing.

## Folder Structure

```
drx_nlp_pipeline/
├── models/                  # LLaMA model file (optional, instructions to download)
│   └── llama-2-7b.Q4_K_M.gguf
├── vector_db/               # FAISS index and metadata, includes db_handler.py for vector search
│   ├── db_handler.py
│   ├── drx_index.faiss
│   └── metadata.pkl
├── extract/                 # Text extraction logic
│   └── extractor.py
├── chunking/                # Tokenization and chunking logic
│   └── tokenizer.py
├── embedding/               # Embedding logic
│   └── embedder.py
├── rag/                     # RAG question-answering system
│   └── qa_system.py
├── translate/               # Translation system
│   └── translator.py
├── summarize/               # Summarization system
│   └── summarizer.py
├── utils/                   # Utility functions
│   └── logger.py
├── data/                    # Folder for input documents
│   └── raw/
├── main.py                  # Main script to run the pipeline
├── requirements.txt         # Python dependencies
├── README.md                
└── .gitignore               
```

## Installation

1. **Clone the repository**:
    ```bash
    git clone <repository_url>
    cd drx_nlp_pipeline
    ```

2. **Create a virtual environment** (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Download the LLaMA model**:
   - The LLaMA model `llama-2-7b.Q4_K_M.gguf` is **too large to be included in the repository**.
   - **Follow these steps** to download the model:
     - Visit [LLaMA's official website or download page](https://example.com/download-link) for the model.
     - Download the model and place the `llama-2-7b.Q4_K_M.gguf` file inside the `models/` folder.

5. **Vector Database Setup**:
   - The **FAISS index** and **metadata** are pre-built and included in the `vector_db/` folder. The `db_handler.py` file in the `vector_db/` folder handles the loading and querying of the FAISS index for efficient similarity-based retrieval.

## Running the Pipeline

To process documents and answer questions, run the following:

```bash
python main.py
```

This will:
- Extract text from files in `data/raw/`
- Tokenize and chunk the content
- Generate embeddings
- Use the RAG system for answering questions

## Extra Credits

- The pipeline **supports CSV**, **Excel** (`.xlsx`, `.xls`, `.xlsm`), **PDF**, and **DOCX** files.
- Handles potential errors in file processing (e.g., unsupported file headers or footers in Excel files).

## Evaluation

The RAG system leverages **FAISS** for fast retrieval and **LLaMA** for contextual generation. Summarization and translation modules are built for multi-lingual support. The system is designed to be modular and extendable.

## Contributing

Feel free to fork, submit issues, or send pull requests for improvements. Contributions are welcome!

---

### Key Points to Highlight

- **Multiple file formats** supported (CSV, DOCX, PDF, Excel).
- **Fast, efficient search** with FAISS and **context-aware answers** using LLaMA.
- **Chunking and tokenization** to handle large documents efficiently.
- **Multilingual capabilities** (Translation and Summarization).
- **Modular design**: Easily extendable for new features or models.
- **Vector database**: Efficient handling of vector searches using FAISS.