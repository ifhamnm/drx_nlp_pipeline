from transformers import pipeline
from utils.logger import log_tokens_per_second
import torch

# Check if CUDA (GPU) is available
device = 0 if torch.cuda.is_available() else -1

# Initialize the summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device)


def summarize_text(text, max_length=200, min_length=30):
    """
    Summarize the input text using a pretrained BART model.

    Args:
        text (str): The text to be summarized.
        max_length (int): Maximum length of the summary.
        min_length (int): Minimum length of the summary.

    Returns:
        str: The summarized version of the text.
    """
    with log_tokens_per_second("📝 Summarization"):
        # Check if the text is too short for summarization
        if len(text.split()) <= min_length:
            return text  # Return the original text if it's too short to summarize

        # Check if text length exceeds the model's max token limit (BART's max is 1024 tokens)
        word_count = len(text.split())

        # If the text is too large (more than 1000 words), we split it into chunks
        if word_count > 1000:
            words = text.split()
            chunk_size = 1000  # Max words per chunk
            chunks = [words[i:i + chunk_size] for i in range(0, len(words), chunk_size)]

            # Summarize each chunk separately
            summaries = []
            for chunk in chunks:
                chunk_text = " ".join(chunk)
                # Skip very short chunks that don't need summarization
                if len(chunk_text.split()) < min_length:
                    continue
                try:
                    summary = summarizer(chunk_text, max_length=max_length, min_length=min_length, do_sample=False)
                    summaries.append(summary[0]['summary_text'])
                except Exception as e:
                    print(f"Error summarizing chunk: {e}")
                    summaries.append(chunk_text)  # Return chunk text in case of failure

            # Combine all summaries and return
            return " ".join(summaries)
        else:
            # If the text is already small enough, summarize it directly
            try:
                summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
                return summary[0]['summary_text']
            except Exception as e:
                print(f"Error summarizing text: {e}")
                return text  # Return the original text if summarization fails
