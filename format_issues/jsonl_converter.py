import json

# Input and output file paths
input_file = "/home2/naga.sai/selfcheckgpt/val/mushroom.en-val.v2_results.jsonl"   # File containing JSON array
output_file = "/home2/naga.sai/selfcheckgpt/val/mushroom.en-val.v2_results_2.jsonl" # Output JSONL file

# Load JSON array from file
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)  # Load the JSON array

# Convert to JSONL format
with open(output_file, "w", encoding="utf-8") as f:
    for obj in data:
        f.write(json.dumps(obj) + "\n")  # Write each object as a separate line
