import os
import PyPDF2
import docx
import tiktoken
import pandas as pd

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page_num in range(len(reader.pages)):
            text += reader.pages[page_num].extract_text()
    return text

# Function to extract text from a DOCX file
def extract_text_from_docx(docx_path):
    doc = docx.Document(docx_path)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

# Function to extract text from an Excel file
def extract_text_from_excel(excel_path):
    dfs = pd.read_excel(excel_path, sheet_name=None)
    text = ""
    for sheet_name, df in dfs.items():
        text += f"Sheet: {sheet_name}\n"
        text += df.astype(str).apply(lambda x: ' | '.join(x), axis=1).str.cat(sep='\n') + "\n"
    return text

# Function to read files from a folder and extract text
def read_documents_from_folder(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            text = extract_text_from_pdf(file_path)
            documents.append({"file": filename, "text": text})
        elif filename.endswith(".docx"):
            file_path = os.path.join(folder_path, filename)
            text = extract_text_from_docx(file_path)
            documents.append({"file": filename, "text": text})
        elif filename.endswith((".xlsx", ".xls", ".xlsm")):
            file_path = os.path.join(folder_path, filename)
            text = extract_text_from_excel(file_path)
            documents.append({"file": filename, "text": text})
    return documents

# Function to chunk documents into smaller parts using tiktoken
def chunk_documents(documents, max_tokens=500):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    all_chunks = []

    for doc in documents:
        text = doc['text']
        tokens = tokenizer.encode(text)

        for i in range(0, len(tokens), max_tokens):
            chunk_tokens = tokens[i:i + max_tokens]
            chunk_text = tokenizer.decode(chunk_tokens)

            all_chunks.append({
                "file": doc['file'],
                "chunk": i // max_tokens + 1,
                "text": chunk_text
            })

    return all_chunks

# Folder where the documents are stored
folder_path = "data/raw"

# Read documents from the folder
documents = read_documents_from_folder(folder_path)

# Chunk the documents
chunks = chunk_documents(documents, max_tokens=500)

# Display the first 200 characters of each chunk for preview
for chunk in chunks:
    print(f"File: {chunk['file']}, Chunk: {chunk['chunk']}")
    print(chunk['text'][:200])  # Preview first 200 characters of the chunk
    print("-" * 40)
