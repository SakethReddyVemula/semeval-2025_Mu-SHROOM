import torch
import spacy

# Load the SpaCy language model
nlp = spacy.load("en_core_web_sm")  # Ensure you have this model installed

# passage = "Did the Scotland team debut in the UEFA Euro 1996 qualifying phase?"
passage = "Yes, Scotland made their debut in the UEFA Euro 1996 qualifying phase. This was their first appearance in a European Championship qualifying campaign since the inception of the UEFA European Football Championship in 1960. Scotland finished third in their group behind England and Switzerland, missing out on qualification for the tournament."
sentences = [sent.text.strip() for sent in nlp(passage).sents] # spacy sentence tokenization
print(sentences)

sample1 = "Yes, the Scotland national football team participated in the UEFA Euro 1996 qualifying phase. Scotland had been part of previous European Championship qualifying campaigns, so their involvement in Euro 1996 qualifiers was not a debut. They successfully qualified for the tournament, finishing second in Group 8 behind Russia. Scotland competed in the final tournament held in England, marking their first appearance in the European Championship since Euro 1992."
sample2 = "No, the Scotland national football team did not debut in the UEFA Euro 1996 qualifying phase. Scotland has a long football history and has participated in international tournaments since the early 20th century. Their debut in the UEFA European Championship qualifying phase dates back to earlier editions of the tournament. Specifically, Scotland first participated in the qualifying phase for UEFA Euro 1968."
sample3 = "Yes, the Scotland national football team participated in the **UEFA Euro 1996 qualifying phase**. However, it was not their debut in the UEFA European Championship qualifiers. Scotland had been competing in UEFA Euro qualifying phases since the tournament's inception in 1960. For Euro 1996, Scotland was placed in **Group 8** of the qualifiers, alongside Russia, Greece, Finland, San Marino, and the Faroe Islands. They finished second in their group, behind Russia, and qualified for the main tournament held in England. In the **Euro 1996 tournament**, Scotland competed in Group A alongside England, the Netherlands, and Switzerland but did not progress beyond the group stage."
sample4 = "Yes, Scotland's national football team participated in the **UEFA Euro 1996 qualifying phase**. However, it was not their debut in a UEFA European Championship qualifying campaign. Scotland had been competing in UEFA European Championship qualifiers since the 1968 edition. In the UEFA Euro 1996 qualifiers, Scotland was placed in **Group 8**, alongside Russia, Greece, Finland, San Marino, and the Faroe Islands. They successfully qualified for the tournament, finishing second in their group behind Russia. Scotland went on to compete in the main tournament held in England."
sample5 = "No, the Scotland team did not debut in the UEFA Euro 1996 qualifying phase. Scotland had participated in previous UEFA European Championship qualifiers before Euro 1996. Their first appearance in a European Championship qualifying phase was during the qualifiers for **Euro 1964**. By the time of UEFA Euro 1996, Scotland was an established participant in international tournaments and their qualifying campaigns. They successfully qualified for Euro 1996, held in England, where they competed in the group stage."
sample6 = "Yes, the Scotland national football team participated in the UEFA Euro 1996 qualifying phase. However, it was not their debut in a UEFA Euro qualification campaign. Scotland had been competing in UEFA European Championship qualifiers since the tournament's inception in 1960.  For Euro 1996, they were placed in Group 8 during the qualifying phase and managed to secure qualification for the final tournament held in England. They advanced by finishing second in their group behind Russia."

# Option1: open-source model
from selfcheckgpt.modeling_selfcheck import SelfCheckNLI
# llm_model = "mistralai/Mistral-7B-Instruct-v0.2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
llm_model = "meta-llama/Llama-3.2-1B"
selfcheck_nli = SelfCheckNLI(device=device) # set device to 'cuda' if GPU is available

sent_scores_nli = selfcheck_nli.predict(
    sentences = sentences,                          # list of sentences
    sampled_passages = [sample1, sample2, sample3], # list of sampled passages
)
print(sent_scores_nli)
# [0.99628643 0.74696505 0.9960837 ] -- based on the example above