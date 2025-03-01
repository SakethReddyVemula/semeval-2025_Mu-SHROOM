import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
from collections import Counter
import math
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer
import torch
import torch.nn.functional as F
import spacy

@dataclass
class Span:
    start: int
    end: int
    text: str

@dataclass
class SpanScore:
    start: int
    end: int
    probability: float

def create_gaussian_window(window_size: int, sigma: float = 1.0) -> np.ndarray:
    """Create a Gaussian window for smoothing."""
    x = np.arange(window_size) - (window_size - 1) / 2
    gaussian = np.exp(-(x**2) / (2 * sigma**2))
    return gaussian / gaussian.sum()  # Normalize

class EnhancedEntropyHallucinationDetector:
    def __init__(
        self,
        window_size: int = 5,  # Reduced window size for finer granularity
        stride: int = 2,       # Reduced stride for better overlap
        entropy_threshold: float = 0.6,  # Increased threshold for more precision
        min_span_length: int = 3,  # Minimum span length to consider
        boundary_threshold: float = 0.3  # Threshold for boundary detection
    ):
        self.window_size = window_size
        self.stride = stride
        self.entropy_threshold = entropy_threshold
        self.min_span_length = min_span_length
        self.boundary_threshold = boundary_threshold
        self.encoder = SentenceTransformer('all-mpnet-base-v2')
        self.nlp = spacy.load('en_core_web_sm')

    def _get_sliding_windows(self, text: str) -> List[Span]:
        """Generate sliding windows over the text."""
        windows = []
        for i in range(0, len(text), self.stride):
            end = min(i + self.window_size, len(text))
            windows.append(Span(i, end, text[i:end]))
            if end == len(text):
                break
        return windows

    def _find_matching_spans(self, target_span: Span, response: str) -> List[str]:
        """Find similar spans in response using sequence matching."""
        matches = []
        matcher = SequenceMatcher(None, target_span.text, response)
        for match in matcher.get_matching_blocks():
            if match.size >= min(3, len(target_span.text)):
                matched_text = response[match.b:match.b + match.size]
                matches.append(matched_text)
        return matches

    def _calculate_semantic_entropy(self, target_span: str, matching_spans: List[str]) -> float:
        """Calculate entropy based on semantic similarity."""
        if not matching_spans:
            return 1.0

        # Encode target and matching spans
        target_emb = self.encoder.encode([target_span])[0]
        match_embs = self.encoder.encode(matching_spans)

        # Calculate cosine similarities
        similarities = F.cosine_similarity(
            torch.tensor(target_emb).unsqueeze(0),
            torch.tensor(match_embs),
            dim=1
        )

        # Convert similarities to probabilities
        probs = F.softmax(similarities, dim=0)
        
        # Calculate entropy
        entropy = -torch.sum(probs * torch.log2(probs + 1e-10))
        return min(entropy.item(), 1.0)

    def _calculate_lexical_entropy(self, matching_spans: List[str]) -> float:
        """Calculate entropy based on lexical variations."""
        if not matching_spans:
            return 1.0

        # Count frequency of each unique span
        freq = Counter(matching_spans)
        total = sum(freq.values())
        probs = [count/total for count in freq.values()]
        
        # Calculate Shannon entropy
        entropy = -sum(p * math.log2(p) for p in probs)
        max_entropy = math.log2(len(matching_spans))
        
        return min(entropy/max_entropy if max_entropy > 0 else 1.0, 1.0)

    def _calculate_frequency_score(self, num_matches: int, total_responses: int) -> float:
        """Calculate score based on how many responses contain matching information."""
        return 1.0 - (num_matches / total_responses)
        
    def _get_linguistic_boundaries(self, text: str) -> List[int]:
        """Get linguistic boundary positions (tokens, phrases, entities)."""
        doc = self.nlp(text)
        boundaries = set()
        
        # Add token boundaries
        for token in doc:
            boundaries.add(token.idx)
            boundaries.add(token.idx + len(token.text))
        
        # Add noun phrase boundaries
        for np in doc.noun_chunks:
            boundaries.add(np.start_char)
            boundaries.add(np.end_char)
            
        # Add named entity boundaries
        for ent in doc.ents:
            boundaries.add(ent.start_char)
            boundaries.add(ent.end_char)
            
        return sorted(list(boundaries))

    def _align_span_to_boundaries(
        self,
        start: int,
        end: int,
        boundaries: List[int],
        scores: np.ndarray
    ) -> Tuple[int, int]:
        """Align span boundaries to nearest linguistic boundaries."""
        # Find nearest linguistic boundaries
        start_candidates = [b for b in boundaries if abs(b - start) <= self.window_size]
        end_candidates = [b for b in boundaries if abs(b - end) <= self.window_size]
        
        if not start_candidates or not end_candidates:
            return start, end
            
        # Choose best boundaries based on score gradient
        best_start = start
        best_end = end
        max_gradient = 0
        
        for s in start_candidates:
            for e in end_candidates:
                if e - s < self.min_span_length:
                    continue
                    
                # Calculate score gradient at boundaries
                start_gradient = abs(scores[min(s+1, len(scores)-1)] - scores[max(s-1, 0)])
                end_gradient = abs(scores[min(e+1, len(scores)-1)] - scores[max(e-1, 0)])
                avg_gradient = (start_gradient + end_gradient) / 2
                
                if avg_gradient > max_gradient:
                    max_gradient = avg_gradient
                    best_start = s
                    best_end = e
                    
        return best_start, best_end

    def _refine_spans(
        self,
        spans: List[Dict[str, float]],
        text: str,
        char_scores: np.ndarray
    ) -> List[Dict[str, float]]:
        """Refine span boundaries using linguistic information and score patterns."""
        boundaries = self._get_linguistic_boundaries(text)
        refined_spans = []
        
        for span in spans:
            # Skip very short spans
            if span['end'] - span['start'] < self.min_span_length:
                continue
                
            # Align to linguistic boundaries
            start, end = self._align_span_to_boundaries(
                span['start'],
                span['end'],
                boundaries,
                char_scores
            )
            
            # Calculate confidence score for the refined span
            span_scores = char_scores[start:end]
            confidence = float(np.mean(span_scores))
            
            # Only keep spans with high enough confidence
            if confidence >= self.entropy_threshold:
                refined_spans.append({
                    'start': start,
                    'end': end,
                    'probability': confidence
                })
        
        # Merge overlapping spans
        refined_spans = self._merge_overlapping_spans(refined_spans)
        return refined_spans

    def _merge_overlapping_spans(
        self,
        spans: List[Dict[str, float]]
    ) -> List[Dict[str, float]]:
        """Merge overlapping spans using score-weighted boundaries."""
        if not spans:
            return spans
            
        # Sort spans by start position
        sorted_spans = sorted(spans, key=lambda x: (x['start'], x['end']))
        merged = []
        current = sorted_spans[0]
        
        for next_span in sorted_spans[1:]:
            if current['end'] >= next_span['start']:
                # Weighted average for probability
                overlap_len = current['end'] - next_span['start']
                total_len = max(current['end'], next_span['end']) - min(current['start'], next_span['start'])
                weight1 = (current['end'] - current['start']) / total_len
                weight2 = (next_span['end'] - next_span['start']) / total_len
                
                current = {
                    'start': min(current['start'], next_span['start']),
                    'end': max(current['end'], next_span['end']),
                    'probability': (current['probability'] * weight1 + 
                                  next_span['probability'] * weight2)
                }
            else:
                merged.append(current)
                current = next_span
        
        merged.append(current)
        return merged

    def detect_hallucinations(
        self,
        model_output: str,
        responses: List[str]
    ) -> Tuple[List[Tuple[int, int]], List[Dict[str, float]]]:
        """
        Detect hallucination spans with improved boundary precision.
        """
        # Calculate initial character-level scores
        char_scores = np.zeros(len(model_output))
        char_counts = np.zeros(len(model_output))
        
        # Get sliding windows
        windows = self._get_sliding_windows(model_output)
        
        # Calculate scores for each window
        for window in windows:
            matches = []
            for response in responses:
                matches.extend(self._find_matching_spans(window, response))
            
            if matches:
                semantic_entropy = self._calculate_semantic_entropy(window.text, matches)
                lexical_entropy = self._calculate_lexical_entropy(matches)
                frequency_score = 1.0 - (len(matches) / len(responses))
                
                score = (semantic_entropy * 0.4 + 
                        lexical_entropy * 0.4 + 
                        frequency_score * 0.2)
            else:
                score = 1.0
            
            char_scores[window.start:window.end] += score
            char_counts[window.start:window.end] += 1
        
        # Average the scores
        char_scores = np.divide(
            char_scores,
            char_counts,
            out=np.zeros_like(char_scores),
            where=char_counts != 0
        )
        
        # Apply Gaussian smoothing to reduce noise
        window_size = 3
        gaussian_window = create_gaussian_window(window_size)
        smoothed_scores = np.convolve(
            char_scores,
            gaussian_window,
            mode='same'
        )
        
        # Find initial spans using score patterns
        spans = []
        in_span = False
        start_idx = 0
        
        for i in range(1, len(smoothed_scores)):
            if not in_span and smoothed_scores[i] >= self.entropy_threshold:
                in_span = True
                start_idx = i
            elif in_span and smoothed_scores[i] < self.entropy_threshold:
                in_span = False
                if i - start_idx >= self.min_span_length:
                    spans.append({
                        'start': start_idx,
                        'end': i,
                        'probability': float(np.mean(smoothed_scores[start_idx:i]))
                    })
        
        # Add final span if needed
        if in_span and len(smoothed_scores) - start_idx >= self.min_span_length:
            spans.append({
                'start': start_idx,
                'end': len(smoothed_scores),
                'probability': float(np.mean(smoothed_scores[start_idx:]))
            })
        
        # Refine spans using linguistic boundaries
        refined_spans = self._refine_spans(spans, model_output, smoothed_scores)
        
        # Generate hard labels from refined spans
        hard_labels = [
            (span['start'], span['end'])
            for span in refined_spans
            if span['probability'] >= self.entropy_threshold
        ]
        
        return hard_labels, refined_spans

# Example usage:
if __name__ == "__main__":
    detector = EnhancedEntropyHallucinationDetector(
        window_size=5,
        stride=2,
        entropy_threshold=0.5,
        min_span_length=3,
        boundary_threshold=0.3
    )
    model_output = "Petra van Stoveren won a silver medal in the 2008 Summer Olympics in Beijing, China."
    responses = [
        "Petra van Staversen won a gold medall at the 2012 Summer Olympics in London. She won the gold medal in the women's 10km open water swimming event.",
        "Petra van Staversen won a gold in the 200m freestyle at the 1996 Summer Olympics.",
        "Petra van Staversen won a gold medall in the 200m breaststroke event at the 1996 Summer Olympics.",
        "Petra van Staversen won a gold in the 200m breaststroke event at the 2016 Summer Olympics.",
        "Petra van Staversen is a Dutch rower who won a gold medals at the 2000 Sydney Olympics. She won a silver medal at the same games. She also won a bronze medal at 2004 Athens Olympics.",
        "Petra van Staversen won a gold medall in the 2012 Summer Olympics in London. She won the gold medal in the women's 10km open water swimming event.",
        "Petra van Staversen won a gold in the 200m freestyle at the 1996 Summer Olympics.",
        "Petra van Staversen won a gold in the 400 meters hurdles at the 2000 Summer Olympics.",
        "Petra van Staversen won a gold medall in the women's 400 meters hurdles at the 2000 Sydney Olympics.",
        "Petra van Staversen won a gold in the 400 meters hurdles at the 2000 Sydney Olympics.",
        "Petra van Staversen won a gold medall in the women's 400 meters hurdles at the 2000 Sydney Olympics.",
        "Petra van Staversen won a gold medall in the 200m breaststroke event at the 2012 Summer Olympics.",
        "Petra van Staversen won a gold medall in the women's 400 meters hurdles at the 2000 Sydney Olympics.",
        "Petra van Staversen won a gold medall in the women's 400 meters hurdles at the 2000 Sydney Olympics.",
        "Petra van Staversen won a gold in the 200m freestyle at the 1996 Summer Olympics.",
        "Petra van Staversen won a gold medall in the 200m breaststroke event at the 1992 Summer Olympics.",
        "Petra van Staversen won a gold medall at the 2000 Sydney Olympics in the women's 400 meters hurdles event.",
        "Petra van Staversen won a gold medall in the 200m breaststroke event at the 2012 Summer Olympics.",
        "Petra van Staversen won a gold medall in the 200m breaststroke event at the 2012 Summer Olympics.",
        "Petra van Staversen won a gold in the women's 400 meters hurdles at the 2000 Sydney Olympics."
    ]
        
    hard_labels, soft_labels = detector.detect_hallucinations(model_output, responses)
    print("Hard Labels:", hard_labels)
    print("Soft Labels:", soft_labels)