import json
from typing import List, Dict
import sys

def load_json(file_path: str) -> List[Dict]:
    """
    Load JSON file and return its contents.
    Handles both single-line JSONs and JSONs with one object per line.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        
        try:
            # First try to parse as a regular JSON
            return json.loads(content)
        except json.JSONDecodeError:
            # If that fails, try to parse as JSON Lines (one JSON object per line)
            try:
                return [json.loads(line.strip()) for line in content.split('\n') if line.strip()]
            except json.JSONDecodeError as e:
                print(f"Error loading {file_path}:")
                print(f"Neither standard JSON nor JSON Lines format could be parsed")
                print(f"First few characters of the file: {content[:100]}...")
                raise e

# def merge_json_files(val_path: str, scores_path: str, responses_path: str, output_path: str) -> None:
def merge_json_files(val_path: str, responses_path: str, output_path: str) -> None:
    """
    Merge three JSON files based on matching 'id' and 'lang' fields.
    
    Args:
        val_path: Path to the first JSON file (val.json)
        # scores_path: Path to the second JSON file (scores.json)
        responses_path: Path to the third JSON file (responses.json)
        output_path: Path where the merged JSON will be saved
    """
    try:
        # Load JSON files
        print(f"Loading {val_path}...")
        val_data = load_json(val_path)
        # print(f"Loading {scores_path}...")
        # scores_data = load_json(scores_path)
        print(f"Loading {responses_path}...")
        responses_data = load_json(responses_path)
        
        # Ensure data is in list format
        if not isinstance(val_data, list):
            val_data = [val_data]
        # if not isinstance(scores_data, list):
        #     scores_data = [scores_data]
        if not isinstance(responses_data, list):
            responses_data = [responses_data]
        
        # Create lookup dictionaries for scores and responses
        # scores_lookup = {(item['id'], item['lang']): item for item in scores_data}
        responses_lookup = {(item['id'], item['lang']): item for item in responses_data}
        
        # Merge the data
        merged_data = []
        for val_item in val_data:
            key = (val_item['id'], val_item['lang'])
            merged_item = val_item.copy()
            
            # # Add score data if available
            # if key in scores_lookup:
            #     merged_item.update({
            #         'sent_scores_nli': scores_lookup[key]['sent_scores_nli']
            #     })
            
            # Add response data if available
            if key in responses_lookup:
                merged_item.update({
                    'responses': responses_lookup[key]['responses']
                })
            
            merged_data.append(merged_item)
        
        # Save the merged data
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, ensure_ascii=False, indent=2)
        
        # Print summary
        print(f"\nSummary:")
        print(f"Total items in val.json: {len(val_data)}")
        # print(f"Total items in scores.json: {len(scores_data)}")
        print(f"Total items in responses.json: {len(responses_data)}")
        print(f"Total items in merged output: {len(merged_data)}")
        
    except FileNotFoundError as e:
        print(f"Error: Could not find file {e.filename}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: JSON parsing failed")
        print(f"Error details: {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        sys.exit(1)

# Example usage
if __name__ == "__main__":
    LANG = "fr"
    val_path = f"/media/saketh/New Volume/semeval-2025_Mu-SHROOM/test-unlabeled/v1/mushroom.{LANG}-tst.v1.jsonl"
    # scores_path = f"/media/saketh/New Volume/semeval-2025_Mu-SHROOM/results/{LANG}_scores.jsonl"
    responses_path = f"/media/saketh/New Volume/semeval-2025_Mu-SHROOM/tst/mushroom.{LANG}-tst.v1_responses.jsonl"
    output_path = f"/media/saketh/New Volume/semeval-2025_Mu-SHROOM/tst/mushroom.{LANG}-tst.v1_results.jsonl"
    
    # merge_json_files(val_path, scores_path, responses_path, output_path)
    merge_json_files(val_path, responses_path, output_path)
