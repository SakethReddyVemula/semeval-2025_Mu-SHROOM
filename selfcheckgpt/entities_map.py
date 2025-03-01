import spacy
import torch
from typing import List, Dict, Tuple

class HallucinationDetector:
    def __init__(self, nlp_model='en_core_web_sm'):
        # Load SpaCy model
        self.nlp = spacy.load(nlp_model)
        
        # Setup SelfCheckNLI (assuming the import and setup from the original code)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        from selfcheckgpt.modeling_selfcheck import SelfCheckNLI
        self.selfcheck_nli = SelfCheckNLI(device=self.device)
    
    def detect_hallucinations(self, passage: str, sample_passages: List[str]) -> Tuple[List[Tuple[int, int]], List[Dict]]:
        # Tokenize sentences
        sentences = [sent.text.strip() for sent in self.nlp(passage).sents]
        
        # Get sentence-level hallucination scores
        sent_scores_nli = self.selfcheck_nli.predict(
            sentences=sentences,
            sampled_passages=sample_passages
        )
        
        # Process named entities
        doc = self.nlp(passage)
        
        # Prepare hard and soft labels
        hard_labels = []
        soft_labels = []
        
        # Process each named entity
        for ent in doc.ents:
            # Find the sentence this entity belongs to
            sent_index = None
            for i, sent in enumerate(doc.sents):
                if ent.start_char >= sent.start_char and ent.end_char <= sent.end_char:
                    sent_index = i
                    break
            
            # If sentence found and has high hallucination score
            if sent_index is not None and sent_scores_nli[sent_index] > 0.5:
                # Hard label: definite hallucination region
                hard_labels.append((ent.start_char, ent.end_char))
                
                # Soft label: probabilistic hallucination
                soft_labels.append({
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'prob': sent_scores_nli[sent_index]
                })
        
        return hard_labels, soft_labels

# Example usage
def main():
    passage = "Yes, Scotland made their debut in the UEFA Euro 1996 qualifying phase. This was their first appearance in a European Championship qualifying campaign since the inception of the UEFA European Football Championship in 1960. Scotland finished third in their group behind England and Switzerland, missing out on qualification for the tournament."
    
    sample_passages = [
        "Yes, the Scotland national football team participated in the UEFA Euro 1996 qualifying phase. Scotland had been part of previous European Championship qualifying campaigns, so their involvement in Euro 1996 qualifiers was not a debut. They successfully qualified for the tournament, finishing second in Group 8 behind Russia. Scotland competed in the final tournament held in England, marking their first appearance in the European Championship since Euro 1992.", 
    "No, the Scotland national football team did not debut in the UEFA Euro 1996 qualifying phase. Scotland has a long football history and has participated in international tournaments since the early 20th century. Their debut in the UEFA European Championship qualifying phase dates back to earlier editions of the tournament. Specifically, Scotland first participated in the qualifying phase for UEFA Euro 1968.",
    "Yes, the Scotland national football team participated in the **UEFA Euro 1996 qualifying phase**. However, it was not their debut in the UEFA European Championship qualifiers. Scotland had been competing in UEFA Euro qualifying phases since the tournament's inception in 1960. For Euro 1996, Scotland was placed in **Group 8** of the qualifiers, alongside Russia, Greece, Finland, San Marino, and the Faroe Islands. They finished second in their group, behind Russia, and qualified for the main tournament held in England. In the **Euro 1996 tournament**, Scotland competed in Group A alongside England, the Netherlands, and Switzerland but did not progress beyond the group stage.",
    "Yes, Scotland's national football team participated in the **UEFA Euro 1996 qualifying phase**. However, it was not their debut in a UEFA European Championship qualifying campaign. Scotland had been competing in UEFA European Championship qualifiers since the 1968 edition. In the UEFA Euro 1996 qualifiers, Scotland was placed in **Group 8**, alongside Russia, Greece, Finland, San Marino, and the Faroe Islands. They successfully qualified for the tournament, finishing second in their group behind Russia. Scotland went on to compete in the main tournament held in England.",
    "No, the Scotland team did not debut in the UEFA Euro 1996 qualifying phase. Scotland had participated in previous UEFA European Championship qualifiers before Euro 1996. Their first appearance in a European Championship qualifying phase was during the qualifiers for **Euro 1964**. By the time of UEFA Euro 1996, Scotland was an established participant in international tournaments and their qualifying campaigns. They successfully qualified for Euro 1996, held in England, where they competed in the group stage.",
    "Yes, the Scotland national football team participated in the UEFA Euro 1996 qualifying phase. However, it was not their debut in a UEFA Euro qualification campaign. Scotland had been competing in UEFA European Championship qualifiers since the tournament's inception in 1960.  For Euro 1996, they were placed in Group 8 during the qualifying phase and managed to secure qualification for the final tournament held in England. They advanced by finishing second in their group behind Russia."
    ]

    
    detector = HallucinationDetector()
    hard_labels, soft_labels = detector.detect_hallucinations(passage, sample_passages)
    
    print("Hard Labels (Definite Hallucination Regions):")
    print(hard_labels)
    print("\nSoft Labels (Probabilistic Hallucination):")
    print(soft_labels)

if __name__ == "__main__":
    main()