import json
import argparse
from typing import List, Dict
from dataclasses import dataclass
from information_entropy import EnhancedEntropyHallucinationDetector

def process_file(input_path: str, test_lang: str) -> List[Dict]:
    """Load and process input file containing model outputs and responses."""
    try:
        with open(input_path, 'r') as f:
            # Read the entire file as a single JSON array
            data = json.load(f)
            
        # Ensure data is a list
        if not isinstance(data, list):
            data = [data]
            
        # Filter for specified language
        return [item for item in data if item.get('lang') == test_lang.upper()]
    except json.JSONDecodeError as e:
        print(f"Error reading JSON file: {e}")
        # Try reading as JSONL format as fallback
        try:
            with open(input_path, 'r') as f:
                data = [json.loads(line.strip()) for line in f if line.strip()]
            return [item for item in data if item.get('lang') == test_lang.upper()]
        except json.JSONDecodeError as e:
            print(f"Error reading JSONL format: {e}")
            raise

def convert_to_soft_labels(spans: List[Dict[str, float]]) -> List[Dict[str, float]]:
    """Convert refined spans to soft labels format."""
    return [
        {
            'start': span['start'],
            'end': span['end'],
            'prob': span['probability']
        }
        for span in spans
    ]

def main(args):
    # Hyperparameters
    WINDOW_SIZE = 3
    STRIDE = 2
    ENTROPY_THRESHOLD = 0.5
    MIN_SPAN_LENGTH = 3
    BOUNDARY_THRESHOLD = 0.3

    # Initialize detector    
    detector = EnhancedEntropyHallucinationDetector(
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        entropy_threshold=ENTROPY_THRESHOLD,
        min_span_length=MIN_SPAN_LENGTH,
        boundary_threshold=BOUNDARY_THRESHOLD
    )

    # Process input file
    data = process_file(args.input_path, args.test_lang)
    
    if not data:
        print(f"No data found for language {args.test_lang}")
        return
        
    # Store results
    hard_labels_all = {}
    soft_labels_all = {}
    predictions_all = []
    
    # Process each sample
    for i, item in enumerate(data):
        print(i)
        model_output = item['model_output_text']
        responses = item['responses']
        
        # Detect hallucinations
        hard_labels, refined_spans = detector.detect_hallucinations(model_output, responses)
        
        # Convert spans to required format
        soft_labels = convert_to_soft_labels(refined_spans)
        
        # Store results
        hard_labels_all[item['id']] = hard_labels
        soft_labels_all[item['id']] = soft_labels
        predictions_all.append({
            'id': item['id'],
            'hard_labels': hard_labels,
            'soft_labels': soft_labels,
            'model_output_text': model_output
        })
    
    # Save results
    output_prefix = f"{args.test_lang}"
    
    with open(f"{output_prefix}-hard_labels.json", 'w') as f:
        json.dump(hard_labels_all, f)
    
    with open(f"{output_prefix}-soft_labels.json", 'w') as f:
        json.dump(soft_labels_all, f)
        
    with open(f"{output_prefix}-pred.jsonl", 'w') as f:
        for pred_dict in predictions_all:
            print(json.dumps(pred_dict), file=f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect hallucinations using Information Entropy method")
    parser.add_argument('--input_path', type=str, required=True, help="Path to input JSON/JSONL file")
    parser.add_argument('--test_lang', type=str, default="en", help="Language code for testing")
    args = parser.parse_args()
    main(args)