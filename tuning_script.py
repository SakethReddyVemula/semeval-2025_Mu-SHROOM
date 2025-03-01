import json
import argparse
import numpy as np
from typing import List, Dict, Tuple
from itertools import product
from dataclasses import dataclass
from information_entropy import EnhancedEntropyHallucinationDetector
from scipy.stats import spearmanr

def process_file(input_path: str, test_lang: str, is_ref: bool = False) -> List[Dict]:
    """Load and process input file containing model outputs and responses."""
    try:
        with open(input_path, 'r') as f:
            data = json.load(f)
            
        if not isinstance(data, list):
            data = [data]
            
        filtered_data = [item for item in data if item.get('lang') == test_lang.upper()]
        
        # Add text_len for reference data
        if is_ref:
            for item in filtered_data:
                item['text_len'] = len(item['model_output_text'])
                
        return filtered_data
    except json.JSONDecodeError as e:
        print(f"Error reading JSON file: {e}")
        try:
            with open(input_path, 'r') as f:
                data = [json.loads(line.strip()) for line in f if line.strip()]
            filtered_data = [item for item in data if item.get('lang') == test_lang.upper()]
            
            # Add text_len for reference data
            if is_ref:
                for item in filtered_data:
                    item['text_len'] = len(item['model_output_text'])
                    
            return filtered_data
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

def score_iou(ref_dict: Dict, pred_dict: Dict) -> float:
    """Compute IoU score between reference and predicted labels."""
    assert ref_dict['id'] == pred_dict['id']
    ref_indices = {idx for span in ref_dict['hard_labels'] for idx in range(*span)}
    pred_indices = {idx for span in pred_dict['hard_labels'] for idx in range(*span)}
    
    if not pred_indices and not ref_indices:
        return 1.0
    
    return len(ref_indices & pred_indices) / len(ref_indices | pred_indices)

def score_cor(ref_dict: Dict, pred_dict: Dict) -> float:
    """Compute correlation score between reference and predicted soft labels."""
    assert ref_dict['id'] == pred_dict['id']
    text_len = ref_dict['text_len']  # Get text length from reference
    
    ref_vec = [0.] * text_len
    pred_vec = [0.] * text_len
    
    for span in ref_dict['soft_labels']:
        for idx in range(span['start'], span['end']):
            ref_vec[idx] = span['prob']
            
    for span in pred_dict['soft_labels']:
        for idx in range(span['start'], span['end']):
            pred_vec[idx] = span['prob']
            
    if len({round(flt, 8) for flt in pred_vec}) == 1 or len({round(flt, 8) for flt in ref_vec}) == 1:
        return float(len({round(flt, 8) for flt in ref_vec}) == len({round(flt, 8) for flt in pred_vec}))
    
    return spearmanr(ref_vec, pred_vec).correlation

def evaluate_predictions(ref_dicts: List[Dict], pred_dicts: List[Dict]) -> Tuple[float, float]:
    """Evaluate predictions using IoU and correlation metrics."""
    assert len(ref_dicts) == len(pred_dicts)
    
    # Sort both lists by ID to ensure matching
    ref_dicts = sorted(ref_dicts, key=lambda x: x['id'])
    pred_dicts = sorted(pred_dicts, key=lambda x: x['id'])
    
    ious = np.array([score_iou(r, p) for r, p in zip(ref_dicts, pred_dicts)])
    cors = np.array([score_cor(r, p) for r, p in zip(ref_dicts, pred_dicts)])
    return ious.mean(), cors.mean()

def run_with_params(
    data: List[Dict],
    ref_data: List[Dict],  # Added reference data parameter
    window_size: int,
    stride: int,
    entropy_threshold: float,
    min_span_length: int,
    boundary_threshold: float
) -> List[Dict]:
    print(f"Window Size: {window_size}\nstride: {stride}\nentropy_threshold: {entropy_threshold}\nmin_span_length: {min_span_length}\nboundary_threshold: {boundary_threshold}\n")

    """Run hallucination detection with specified parameters."""
    detector = EnhancedEntropyHallucinationDetector(
        window_size=window_size,
        stride=stride,
        entropy_threshold=entropy_threshold,
        min_span_length=min_span_length,
        boundary_threshold=boundary_threshold
    )

    # Create a mapping of reference text lengths
    ref_lengths = {item['id']: item['text_len'] for item in ref_data}
    
    predictions = []
    for i, item in enumerate(data):
        if i % 5 == 0:
            print(f"progress: {i}/50")
        hard_labels, refined_spans = detector.detect_hallucinations(
            item['model_output_text'],
            item['responses']
        )
        soft_labels = convert_to_soft_labels(refined_spans)
        predictions.append({
            'id': item['id'],
            'hard_labels': hard_labels,
            'soft_labels': soft_labels,
            'text_len': ref_lengths[item['id']]  # Use text length from reference
        })
    
    return predictions

def grid_search(data: List[Dict], ref_data: List[Dict], param_grid: Dict) -> Tuple[Dict, Dict]:
    """Perform grid search over hyperparameters."""
    best_score = -float('inf')
    best_params = None
    all_results = {}
    
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    
    for values in product(*param_values):
        params = dict(zip(param_names, values))
        
        # Run detection with current parameters
        predictions = run_with_params(data, ref_data, **params)
        
        # Evaluate predictions
        iou_score, cor_score = evaluate_predictions(ref_data, predictions)
        combined_score = (iou_score + cor_score) / 2
        
        # Store results
        all_results[tuple(params.items())] = {
            'iou': iou_score,
            'correlation': cor_score,
            'combined': combined_score
        }
        
        # Update best parameters if needed
        if combined_score > best_score:
            best_score = combined_score
            best_params = params
            
        print(f"Params: {params}")
        print(f"IoU: {iou_score:.4f}, Correlation: {cor_score:.4f}, Combined: {combined_score:.4f}\n")
    
    return best_params, all_results

def main(args):
    # Load data with is_ref parameter
    data = process_file(args.input_path, args.test_lang, is_ref=False)
    ref_data = process_file(args.ref_path, args.test_lang, is_ref=True)
    
    if not data or not ref_data:
        print(f"No data found for language {args.test_lang}")
        return
        
    # Define parameter grid
    param_grid = {
        'window_size': [5],
        # 'stride': [1, 2],
        # 'entropy_threshold': [0.3, 0.5, 0.7],
        # 'min_span_length': [2, 3, 4],
        # 'boundary_threshold': [0.2, 0.3, 0.4]
        'stride': [3],
        'entropy_threshold': [0.5],
        'min_span_length': [3, 4, 5, 6, 7],
        'boundary_threshold': [0.3]
    }
    
    # Perform grid search
    best_params, all_results = grid_search(data, ref_data, param_grid)
    
    # Save results
    output_prefix = f"{args.test_lang}"
    
    with open(f"{output_prefix}-tuning_results.json", 'w') as f:
        json.dump({
            'best_params': best_params,
            'all_results': {str(k): v for k, v in all_results.items()}
        }, f, indent=2)
    
    # Run final prediction with best parameters
    final_predictions = run_with_params(data, ref_data, **best_params)
    
    with open(f"{output_prefix}-best_predictions.jsonl", 'w') as f:
        for pred in final_predictions:
            json.dump(pred, f)
            f.write('\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for hallucination detection")
    parser.add_argument('--input_path', type=str, required=True, help="Path to input JSON/JSONL file")
    parser.add_argument('--ref_path', type=str, required=True, help="Path to reference JSON/JSONL file")
    parser.add_argument('--test_lang', type=str, default="en", help="Language code for testing")
    args = parser.parse_args()
    main(args)