import openai
import json
import glob
from tqdm import tqdm
import os
import time

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")  # Or replace with your actual key: "your-api-key"

def generate_response(api_key, prompt, max_length=256, num_responses=1, temperature=0.7):
    """Generate responses using OpenAI's GPT-4 API."""
    openai.api_key = api_key  # Ensure the key is set for each call
    
    responses = []
    for _ in range(num_responses):
        while True:
            try:
                completion = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_length,
                    temperature=temperature
                )
                response_text = completion['choices'][0]['message']['content']
                responses.append(response_text.strip())
                break
            except openai.error.RateLimitError:
                print("Rate limit exceeded. Retrying in 5 seconds...")
                time.sleep(5)  # Wait and retry in case of rate limit errors
    
    return responses

def process_files(api_key, dataset, input_dir, language, output_file, num_responses=20):
    """Process all JSON files in the input directory and generate responses."""
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

            # Generate multiple responses
            responses = generate_response(api_key, input_question, num_responses=num_responses)
            response_objects = [{
                'model_output_text': response,
                'hard_labels': [],  # Empty as these need to be annotated
                'soft_labels': []   # Empty as these need to be annotated
            } for response in responses]
            
            # Create new data point containing all responses
            new_data = {
                'id': data['id'],
                'lang': data['lang'],
                'model_input': input_question,
                'model_id': "gpt-4",
                'responses': response_objects
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
    
    parser = argparse.ArgumentParser(description='Generate GPT-4 responses from input files')
    parser.add_argument('--api_key', type=str, required=True,
                        help='Your OpenAI API key')
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
    
    process_files(args.api_key, args.dataset, args.input_dir, args.language, args.output_file, args.num_responses)