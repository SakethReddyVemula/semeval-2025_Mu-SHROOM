import os
import torch
import json
import argparse
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from datasets import load_dataset, DatasetDict
import evaluate
import wandb

LABEL_LIST = [0, 1]
LANGS = ['ar', 'de', 'en', 'es', 'fi', 'fr', 'hi', 'it', 'sv', 'zh']
MODEL_NAME = 'FacebookAI/xlm-roberta-base'

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def tokenize_and_map_labels(examples, tokenizer):
    tokenized_inputs = tokenizer(examples['model_output_text'], return_offsets_mapping=True, padding=True, truncation=True)
    offset_mappings = tokenized_inputs['offset_mapping']
    all_labels = examples['hard_labels']
    tok_labels_batch = []
    for batch_idx in range(len(offset_mappings)):
        offset_mapping = offset_mappings[batch_idx]
        hard_labels = all_labels[batch_idx]
        tok_labels = [0] * len(offset_mapping)
        for idx, start_end in enumerate(offset_mapping):
            start = start_end[0]
            end = start_end[1]
            for (label_start, label_end) in hard_labels:
                if start >= label_start and end <= label_end:
                    tok_labels[idx] = 1
        tok_labels_batch.append(tok_labels)
    
    return {
        'input_ids': tokenized_inputs['input_ids'],
        'attention_mask': tokenized_inputs['attention_mask'],
        'labels': tok_labels_batch
    }

def train_model(local_rank, test_lang, data_path, output_dir):
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    setup(local_rank, world_size)
    
    torch.cuda.set_device(local_rank)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model = model.to(local_rank)
    model = DistributedDataParallel(model, device_ids=[local_rank])

    data_files = {
        'train': [f'{data_path}/mushroom.{lang}-val.v2.jsonl' for lang in LANGS if lang != test_lang],
        'validation': f'{data_path}/mushroom.{test_lang}-val.v2.jsonl'
    }
    dataset = load_dataset('json', data_files=data_files)
    
    tokenized_datasets = dataset.map(
        lambda x: tokenize_and_map_labels(x, tokenizer), 
        batched=True, 
        remove_columns=dataset['train'].column_names
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy='epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=5,
        weight_decay=0.01,
        report_to='wandb',
        run_name=f'distributed-{test_lang}',
        remove_unused_columns=False
    )

    metric = evaluate.load('seqeval')

    def compute_metrics(p):
        predictions, labels = p
        predictions = torch.argmax(torch.tensor(predictions), dim=2)
        true_labels = [[LABEL_LIST[l] for l in label if l != -100] for label in labels]
        true_predictions = [
            [LABEL_LIST[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(output_dir)
    
    cleanup()

def test_model(test_lang, model_path, data_path):
    # [Rest of the test_model function remains the same as in previous version]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    test_dataset = load_dataset('json', data_files={'test': f'{data_path}/mushroom.{test_lang}-val.v2.jsonl'})['test']
    
    inputs = tokenizer(test_dataset['model_output_text'], padding=True, truncation=True, return_offsets_mapping=True, return_tensors="pt")

    model.eval()
    with torch.no_grad():
        outputs = model(inputs.input_ids)
    preds = torch.argmax(outputs.logits, dim=2)
    probs = F.softmax(outputs.logits, dim=2)
    
    hard_labels_all, soft_labels_all, predictions_all = {}, {}, []
    for i, pred in enumerate(preds):
        hard_labels_sample, soft_labels_sample = [], []
        positive_indices = torch.nonzero(pred == 1, as_tuple=False)
        offset_mapping = inputs['offset_mapping'][i]
        
        for j, offset in enumerate(offset_mapping):
            soft_labels_sample.append({'start': offset[0].item(), 'end': offset[1].item(), 'prob': probs[i][j][1].item()})
            if j in positive_indices:
                hard_labels_sample.append((offset[0].item(), offset[1].item()))
        
        soft_labels_all[test_dataset['id'][i]] = soft_labels_sample
        hard_labels_all[test_dataset['id'][i]] = hard_labels_sample
        predictions_all.append({
            'id': test_dataset['id'][i], 
            'hard_labels': hard_labels_sample, 
            'soft_labels': soft_labels_sample
        })
    
    with open(f"{test_lang}-hard_labels.json", 'w') as f:
        json.dump(hard_labels_all, f)
    with open(f"{test_lang}-soft_labels.json", 'w') as f:
        json.dump(soft_labels_all, f)
    with open(f"{test_lang}-pred.jsonl", 'w') as f:
        for pred_dict in predictions_all:
            print(json.dumps(pred_dict), file=f)

def main(args):
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA devices available")
    
    wandb.init(project=os.getenv('WANDB_PROJECT', 'semeval-2025'))
    
    if args.mode == 'train':
        train_model(
            local_rank=int(os.environ.get('LOCAL_RANK', 0)), 
            test_lang=args.test_lang, 
            data_path=args.data_path, 
            output_dir=args.model_checkpoint
        )
    else:
        test_model(
            test_lang=args.test_lang, 
            model_path=args.model_checkpoint, 
            data_path=args.data_path
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or test the model")
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train')
    parser.add_argument('--data_path', type=str, help="Path to the training data")
    parser.add_argument('--model_checkpoint', type=str, default="./results", help="Path to the trained checkpoint")
    parser.add_argument('--test_lang', type=str, default="en")
    args = parser.parse_args()
    main(args)