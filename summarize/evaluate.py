from rouge_score import rouge_scorer

def evaluate_summary(reference, generated):
    if not reference or not generated:
        raise ValueError("Both reference and generated texts must be non-empty.")

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, generated)
    return scores

