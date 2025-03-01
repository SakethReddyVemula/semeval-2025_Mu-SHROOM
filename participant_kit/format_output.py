import spacy
import numpy as np
from typing import List, Tuple

def annotate_hallucinations(sentences_scores: List[Tuple[str, float]]) -> List[str]:
    """
    Transform sentence-level hallucination scores to word-level annotations.
    
    Args:
        sentences_scores: List of tuples (sentence, hallucination_score)
    
    Returns:
        List of annotated sentences with word-level hallucination scores
    """
    # Load spaCy English model for NLP processing
    nlp = spacy.load("en_core_web_sm")
    
    # Define hallucination score mapping
    def get_word_score(token, sentence_score):
        # Prioritize nouns, proper nouns, numbers, and dates for higher hallucination scores
        if token.pos_ in ['PROPN', 'NOUN', 'NUM'] or token.ent_type_ in ['DATE', 'CARDINAL', 'ORD'] or token.text.lower() in ['yes', 'no', 'true', 'false']:
            return sentence_score  # High likelihood of hallucination
        else:
            return 0.0  # Lower likelihood for other parts of speech

    annotated_sentences = []
    
    for sentence, score in sentences_scores:
        # Process the sentence with spaCy
        doc = nlp(sentence)
        
        # Annotate tokens
        annotated_tokens = []
        for token in doc:
            word_score = get_word_score(token, score)
            if word_score != 0.0:
                annotated_token = f"[HALLUC:{word_score:.2f}]{token.text}[/HALLUC]"
            else:
                annotated_token = f"{token.text}"
            annotated_tokens.append(annotated_token)
        
        # Reconstruct the sentence with annotations
        annotated_sentence = ' '.join(annotated_tokens)
        annotated_sentences.append(annotated_sentence)
    
    return annotated_sentences

# Example usage
sentences_scores = [
    ("Yes, Scotland made their debut in the UEFA Euro 1996 qualifying phase.", 0.99628643),
    ("This was their first appearance in a European Championship qualifying campaign since the inception of the UEFA European Football Championship in 1960.", 0.74696505),
    ("Scotland finished third in their group behind England and Switzerland, missing out on qualification for the tournament.", 0.9960837)
]

annotated_results = annotate_hallucinations(sentences_scores)
for result in annotated_results:
    print(result)