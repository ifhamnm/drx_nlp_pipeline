import os
from extract.extractor import extract_all_files
from chunking.tokenizer import chunk_documents
from embedding.embedder import generate_embeddings
from vector_db.db_handler import build_vector_db, search_vector_db
from rag.qa_system import ask_question
from translate.translator import translate_text
from summarize.summarizer import summarize_text
from utils.logger import log_tokens_per_second
from summarize.evaluate import evaluate_summary


def process_publications(folder_path):
    # Step 1: Extract text from files
    print("📂 Extracting text from publications...")
    documents = extract_all_files(folder_path)

    # Step 2: Chunk the documents into manageable parts
    print("🔪 Tokenizing and chunking documents...")
    chunks = chunk_documents(documents)

    # Step 3: Generate embeddings for the chunks
    print("🔢 Generating embeddings...")
    embedded_chunks = generate_embeddings(chunks)

    # Step 4: Build FAISS vector database
    print("💾 Building FAISS vector database...")
    build_vector_db(embedded_chunks)


def run_qa_system(question):
    print("❓ Answering question using RAG system...")
    answer = ask_question(question)
    return answer


def main():
    # Process publications (adjust the folder path as needed)
    folder_path = "data/raw"  # Set your publication folder here
    process_publications(folder_path)

    # Example Q&A Interaction
    question = "What is the main focus of Dr. X's research?"
    print("\n💬 Q&A:")
    answer = run_qa_system(question)
    print(f"Answer: {answer}")

    # Example Translation
    print("\n🌍 Translation:")
    translated_text = translate_text("Bonjour, je suis un chercheur.", src_lang="fr", target_lang="en")
    print(f"Translated Text: {translated_text}")

    # Example Summarization
    print("\n📝 Summarization:")
    summary = summarize_text(
        "Dr. X's research covers advancements in NLP, including cross-lingual model development and AI ethics.")
    print(f"Summary: {summary}")

    # Example Evaluation
    print("\n📝 Evaluation:")
    reference_text = "Dr. X's research focuses on advancements in NLP, AI ethics, and language models."
    generated_text = summary  # Using the summary as the generated text for this example
    evaluation_scores = evaluate_summary(reference_text, generated_text)
    print(f"Evaluation (ROUGE Scores): {evaluation_scores}")


if __name__ == "__main__":
    main()
