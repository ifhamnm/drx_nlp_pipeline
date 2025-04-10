import os
import pandas as pd
import fitz  # PyMuPDF
import docx

def extract_text_from_pdf(filepath):
    doc = fitz.open(filepath)
    text_data = []
    for page_number in range(len(doc)):
        page = doc.load_page(page_number)
        text = page.get_text()
        text_data.append({
            "file": os.path.basename(filepath),
            "page": page_number + 1,
            "text": text.strip()
        })
    return text_data

def extract_text_from_docx(filepath):
    doc = docx.Document(filepath)
    text = "\n".join([para.text for para in doc.paragraphs])
    return [{
        "file": os.path.basename(filepath),
        "page": 1,
        "text": text.strip()
    }]

def extract_text_from_excel(filepath):
    # Handling multiple Excel formats
    try:
        dfs = pd.read_excel(filepath, sheet_name=None)
    except Exception as e:
        return f"❌ Error reading Excel file: {e}"

    results = []
    for sheet_name, df in dfs.items():
        text = df.astype(str).apply(lambda x: ' | '.join(x), axis=1).str.cat(sep='\n')
        results.append({
            "file": os.path.basename(filepath),
            "page": 1,
            "text": f"Sheet: {sheet_name}\n{text.strip()}"
        })
    return results

def extract_text_from_csv(filepath):
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        return f"❌ Error reading CSV file: {e}"

    text = df.astype(str).apply(lambda x: ' | '.join(x), axis=1).str.cat(sep='\n')
    return [{
        "file": os.path.basename(filepath),
        "page": 1,
        "text": text.strip()
    }]

def extract_all_files(folder_path):
    extracted = []
    for filename in os.listdir(folder_path):
        path = os.path.join(folder_path, filename)
        if filename.endswith(".pdf"):
            extracted.extend(extract_text_from_pdf(path))
        elif filename.endswith(".docx"):
            extracted.extend(extract_text_from_docx(path))
        elif filename.endswith(".csv"):
            extracted.extend(extract_text_from_csv(path))
        elif filename.endswith((".xlsx", ".xls", ".xlsm")):
            extracted.extend(extract_text_from_excel(path))
    return extracted

file_path = 'data/raw'  # Change to the file path you're testing
text = extract_all_files(file_path)
print(text)  # Print the extracted text
