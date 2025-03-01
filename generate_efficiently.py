import json
import glob
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import gc

def load_model(model_name="meta-llama/Llama-3.2-3B-Instruct"):
    """Load the model and tokenizer with memory-efficient configuration."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        pad_token_id=tokenizer.pad_token_id,
        low_cpu_mem_usage=True,
        offload_folder="offload",
        max_memory={0: "6GiB", "cpu": "8GiB"}
    )
    
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_length=64):
    """Generate a single response."""
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                max_new_tokens=max_length,
                num_return_sequences=1,
                temperature=0.1,
                top_p=0.9,
                top_k=50,
                do_sample=True,
                no_repeat_ngram_size=3,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(
            generated_tokens[0][inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        ).strip()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return response
    
    except Exception as e:
        print(f"Error generating response: {str(e)}")
        return ""

def process_files(model_id, dataset, input_dir, language, output_file, num_responses=20):
    """Process files with minimal storage requirements."""
    model, tokenizer = load_model(model_id)
    
    json_files = glob.glob(os.path.join(input_dir, f"mushroom.{language}-{dataset}.v1.jsonl"))
    
    if not json_files:
        raise FileNotFoundError(f"No matching files found in {input_dir}")
    
    json_file = json_files[0]
    
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_filename = f"mushroom.{language}-{dataset}.v1_responses.jsonl"
    output_path = os.path.join(os.path.dirname(output_file), output_filename)
    
    # Process line by line
    with open(json_file, 'r') as f:
        for line_number, line in enumerate(f, 1):
            data = json.loads(line.strip())
            input_question = data['model_input']
            print(f"Processing example {line_number}, id: {data['id']}")
            
            responses = []
            for i in range(num_responses):
                response = generate_response(model, tokenizer, input_question)
                if response:
                    print(f"{i + 1}: {response}")
                    responses.append(response)
                
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Write only essential information
            with open(output_path, 'a') as f:
                minimal_data = {
                    'id': data['id'],
                    'lang': data['lang'],
                    'model_input': input_question,
                    'responses': responses
                }
                f.write(json.dumps(minimal_data) + '\n')
            
            del responses
            gc.collect()

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