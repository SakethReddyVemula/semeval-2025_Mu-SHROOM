import json
import glob
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import os
from datetime import datetime

def load_model(model_name="meta-llama/Llama-3.2-3B-Instruct"):
    """Load the model and tokenizer with proper configuration."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Ensure the tokenizer has padding and EOS tokens set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        pad_token_id=tokenizer.pad_token_id,
    )
    
    # Set the model's padding token ID
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_length=64):
    """Generate a single response from the model."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        generate_outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,  # Changed from max_length to max_new_tokens
            num_return_sequences=1,
            temperature=0.1,  # Increased temperature for more diverse responses
            top_p=0.9,  # Added nucleus sampling
            top_k=50,   # Added top-k sampling
            do_sample=True,
            no_repeat_ngram_size=3,  # Prevent repetition of 3-grams
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,  # Explicitly set EOS token
            return_dict_in_generate=True,
            output_scores=True
        )
    
    # Get the generated token ids
    generated_tokens = generate_outputs.sequences[0]
    
    # Decode only the new tokens (excluding the prompt)
    response = tokenizer.decode(
        generated_tokens[inputs.input_ids.shape[1]:], 
        skip_special_tokens=True
    ).strip()
    
    # Convert token ids to tokens (for the new tokens only)
    tokens = tokenizer.convert_ids_to_tokens(
        generated_tokens[inputs.input_ids.shape[1]:].tolist()
    )
    
    # Get the logits for the generated sequence
    logits = torch.stack(generate_outputs.scores, dim=0)
    logits = logits.cpu().numpy().tolist()
    
    return response, tokens, logits

def process_files(model_id, dataset, input_dir, language, output_file, num_responses=20):
    """Process all JSON files in the input directory and generate responses."""
    model, tokenizer = load_model(model_id)
    
    # Get all JSON files in the input directory
    json_files = glob.glob(os.path.join(input_dir, f"mushroom.{language}-{dataset}.v2.jsonl"))
    
    generated_data = []
    if not json_files:
        raise FileNotFoundError(f"No matching files found in {input_dir}")
    json_file = json_files[0]  # Take the first matching file

    with open(json_file, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            
            # Get the original input question
            input_question = data['model_input']
            print(f"id: {data['id']}\tinput_question: {input_question}\n")

            # Responses
            responses = []

            # Generate multiple responses
            for i in range(num_responses):
                response, tokens, logits = generate_response(model, tokenizer, input_question)
                print(f"response-{i}: {response}")
                responses.append({
                    'model_output_text': response,
                    'model_output_tokens': tokens,
                    'model_output_logits': logits,
                    'hard_labels': [],  # Empty as these need to be annotated
                    'soft_labels': []   # Empty as these need to be annotated
                })
                
            # Create new data point containing all responses
            new_data = {
                'id': data['id'],
                'lang': data['lang'],
                'model_input': input_question,
                'model_id': model_id,
                'responses': responses
            }
            
            generated_data.append(new_data)

    # Save generated data
    output_filename = f"mushroom.{language}-{dataset}.v2_responses.jsonl"
    output_path = os.path.join(os.path.dirname(output_file), output_filename)
    
    with open(output_path, 'w') as f:
        for data in generated_data:
            f.write(json.dumps(data) + '\n')
    
    print(f"Generated responses saved to: {output_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate LLM responses from input files')
    parser.add_argument('--model_id', type=str, default='meta-llama/Llama-3.2-3B-Instruct',
                    help='Model to be used to generate candidate responses')
    parser.add_argument('--dataset', type=str, default='val',
                    help='Train-set / Val-set / Test-set; Args: train, val, test')
    parser.add_argument('--language', type=str, default='en',
                    help='Language of queries')
    parser.add_argument('--input_dir', type=str, default='./val',
                        help='Directory containing input JSON files')
    parser.add_argument('--output_file', type=str, default='./generated_responses.jsonl',
                        help='Output file path')
    parser.add_argument('--num_responses', type=int, default=20,
                        help='Number of responses to generate per input')
    
    args = parser.parse_args()
    
    process_files(args.model_id, args.dataset, args.input_dir, args.language, args.output_file, args.num_responses)