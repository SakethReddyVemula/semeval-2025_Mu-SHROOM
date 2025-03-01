import json
import torch
import spacy
import numpy as np
from selfcheckgpt.modeling_selfcheck import SelfCheckNLI
import argparse
from typing import List, Dict
import os

def load_jsonl(file_path: str) -> List[Dict]:
    """Load data from a JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def get_sentences(text: str, nlp) -> List[str]:
    """Extract sentences from text using SpaCy."""
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]

def numpy_to_python(obj):
    """Convert numpy types to Python native types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, list):
        return [numpy_to_python(item) for item in obj]
    return obj

def process_hallucinations(responses_file: str, val_file: str, output_file: str):
    """Process files and detect hallucinations using SelfCheckGPT."""
    # Initialize SpaCy and SelfCheckGPT
    nlp = spacy.load("en_core_web_sm")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selfcheck_nli = SelfCheckNLI(device=device)
    
    # Load the data
    responses_data = load_jsonl(responses_file)
    val_data = {item['id']: item for item in load_jsonl(val_file)}
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Process each example
    results = []
    for item in responses_data:
        print(f"Processing ID: {item['id']}")
        
        # Get target text from val file
        target_text = val_data[item['id']]['model_output_text']
        target_sentences = get_sentences(target_text, nlp)

        response_scores = []
        try:
            sent_scores = selfcheck_nli.predict(
                sentences=target_sentences,
                sampled_passages=item['responses']
            )
            # Convert numpy array to Python list
            response_scores.append(numpy_to_python(sent_scores))
        except Exception as e:
            print(f"Error processing response: {str(e)}")
            response_scores.append([0.0] * len(target_sentences))
            
        # Create result entry
        result = {
            'id': item['id'],
            'lang': item['lang'],
            'model_input': item['model_input'],
            'sent_scores_nli': response_scores
        }
        
        try:
            # Write to output file immediately to save memory
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result) + '\n')
        except TypeError as e:
            print(f"Error serializing result for ID {item['id']}: {str(e)}")
            continue
        
        results.append(result)
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print(f"Processing complete. Results written to {output_file}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Detect hallucinations in generated responses')
    parser.add_argument('--responses_file', type=str, required=True,
                        help='Path to the responses JSONL file')
    parser.add_argument('--val_file', type=str, required=True,
                        help='Path to the validation JSONL file')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Path to the output JSONL file')
    
    args = parser.parse_args()
    
    process_hallucinations(
        responses_file=args.responses_file,
        val_file=args.val_file,
        output_file=args.output_file
    )
