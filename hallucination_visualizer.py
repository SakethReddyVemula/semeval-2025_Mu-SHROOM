import json
import numpy as np

class HallucinationVisualizer:
    def __init__(self, data):
        """Initialize the visualizer with dataset information"""
        self.data = data
        
    def highlight_hallucinations(self, show_soft_labels=True):
        """Visualize hallucination spans with different highlighting methods"""
        # Extract model output text
        text = self.data['model_output_text']
        
        # Create a character-level annotation array
        annotations = np.zeros(len(text), dtype=float)
        
        # Mark hard label hallucinations
        for start, end in self.data.get('hard_labels', []):
            annotations[start:end] = 1.0
        
        # Process soft labels if available
        if show_soft_labels and 'soft_labels' in self.data:
            for span in self.data.get('soft_labels', []):
                start = span['start']
                end = span['end']
                prob = span.get('prob', 0)
                # Use np.maximum to handle array-wise maximum
                annotations[start:end] = np.maximum(annotations[start:end], prob)
        
        # Create annotated output
        annotated_text = []
        current_annotation = None
        current_span = []
        
        for i, (char, anno) in enumerate(zip(text, annotations)):
            if anno != current_annotation:
                # Save previous span
                if current_span:
                    span_text = ''.join(current_span)
                    if current_annotation > 0:
                        annotated_text.append(f"[HALLUC:{current_annotation:.2f}]{span_text}[/HALLUC]")
                    else:
                        annotated_text.append(span_text)
                
                # Start new span
                current_span = [char]
                current_annotation = anno
            else:
                current_span.append(char)
        
        # Handle the last span
        if current_span:
            span_text = ''.join(current_span)
            if current_annotation > 0:
                annotated_text.append(f"[HALLUC:{current_annotation:.2f}]{span_text}[/HALLUC]")
            else:
                annotated_text.append(span_text)
        
        return ''.join(annotated_text)
    
    def summarize_hallucinations(self):
        """Provide a summary of hallucination spans"""
        summary = {
            'total_text_length': len(self.data['model_output_text']),
            'hard_label_spans': self.data.get('hard_labels', []),
            'soft_label_spans': self.data.get('soft_labels', [])
        }
        
        # Calculate hallucination coverage
        text_length = len(self.data['model_output_text'])
        hard_halluc_length = sum(end - start for start, end in summary['hard_label_spans'])
        summary['hard_hallucination_coverage'] = hard_halluc_length / text_length
        
        # Soft label analysis
        soft_halluc_probs = [span['prob'] for span in summary['soft_label_spans']]
        if soft_halluc_probs:
            summary['avg_soft_halluc_prob'] = np.mean(soft_halluc_probs)
            summary['max_soft_halluc_prob'] = np.max(soft_halluc_probs)
        
        return summary

# Example usage function
def visualize_hallucination_data(data):
    """
    Visualize hallucination data for a single datapoint
    
    :param data: Dictionary containing hallucination data
    :return: Tuple of annotated text and hallucination summary
    """
    visualizer = HallucinationVisualizer(data)
    annotated_text = visualizer.highlight_hallucinations()
    summary = visualizer.summarize_hallucinations()
    
    return annotated_text, summary

# Example data (you can replace this with your actual dataset)
# # Predicted
example_data = {
    "id": 142,
    "lang": "IT",
    "model_input": "Chi ha doppiato in italiano l'attrice Catherine McCormack nel film Il sarto di Panama?",
    "model_output_text": "L'attrice Catherine McCormack è stata doppiata in italiano da Elisa Di Stefano nel film \"Il sarto di Panama\"",
    "model_id": "Qwen\/Qwen2-7B-Instruct",
    "soft_labels": [{"start": 6, "end": 9, "prob": 0.7411115202370769}, {"start": 61, "end": 79, "prob": 0.8330555555555557}, {"start": 97, "end": 101, "prob": 0.7037500000000001}],
    "hard_labels": [[6, 9], [61, 79], [97, 101]]
}

# Ground truth
# example_data = {
#     "id": 142,
#     "lang": "IT",
#     "model_input": "Chi ha doppiato in italiano l'attrice Catherine McCormack nel film Il sarto di Panama?",
#     "model_output_text": "L'attrice Catherine McCormack è stata doppiata in italiano da Elisa Di Stefano nel film \"Il sarto di Panama\"",
#     "model_id": "Qwen\/Qwen2-7B-Instruct",
#     "soft_labels": [{"start":62,"prob":1.0,"end":78}],
#     "hard_labels":[[62,78]]
# }

# Demonstration
annotated_text, summary = visualize_hallucination_data(example_data)
print("Annotated Text:")
print(annotated_text)
print("\nHallucination Summary:")
print(json.dumps(summary, indent=2))