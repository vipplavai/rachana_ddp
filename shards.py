import os
import json
import torch
import torch.distributed as dist
import logging
from datasets import load_dataset
from tokenizers import Tokenizer

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load Hugging Face dataset and split
def load_and_split_dataset():
    dataset = load_dataset('KPrashanth/Telugu_sentences', split='train')
    total_sentences = len(dataset)
    
    # Calculate splits
    train_size = int(0.8 * total_sentences)
    valid_size = int(0.1 * total_sentences)
    test_size = total_sentences - train_size - valid_size

    train_data = dataset.select(range(train_size))
    valid_data = dataset.select(range(train_size, train_size + valid_size))
    test_data = dataset.select(range(train_size + valid_size, total_sentences))

    return train_data, valid_data, test_data

# Save data for each node
def save_data_per_node(train_data, rank, world_size, output_dir):
    total_sentences = len(train_data)
    sentences_per_node = total_sentences // world_size

    start_idx = rank * sentences_per_node
    end_idx = start_idx + sentences_per_node if rank != world_size - 1 else total_sentences

    node_data = train_data.select(range(start_idx, end_idx))
    node_file = os.path.join(output_dir, f'node_{rank}.json')

    node_data.to_json(node_file)
    logging.info(f"Saved {len(node_data)} sentences to {node_file} for rank {rank}")
    
    return node_file

# Tokenize and filter sentences
def tokenize_and_filter_sentences(file_path, tokenizer):
    tokenized_sentences = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                sentence = entry.get('text', '')
                tokens = tokenizer.encode(sentence)
                if len(tokens.ids) <= 256:
                    tokenized_sentences.append({
                        'input_tokens': tokens.tokens,
                        'input_ids': tokens.ids
                    })
            except json.JSONDecodeError as e:
                logging.error(f"Error decoding JSON in file {file_path}: {e}")
                continue

    logging.info(f"Tokenized and filtered {len(tokenized_sentences)} sentences from {file_path}")
    return tokenized_sentences



# Create input-target pairs and save as JSON files
def create_input_target_pairs(tokenized_sentences, output_dir, prefix, chunk_size=3000):
    os.makedirs(output_dir, exist_ok=True)
    num_chunks = len(tokenized_sentences) // chunk_size + (1 if len(tokenized_sentences) % chunk_size > 0 else 0)
    
    for i in range(num_chunks):
        chunk = tokenized_sentences[i*chunk_size:(i+1)*chunk_size]
        input_target_pairs = []

        for sentence in chunk:
            for j in range(1, len(sentence['input_ids'])):
                input_tokens = sentence['input_tokens'][:j]
                target_token = sentence['input_tokens'][j]
                input_ids = sentence['input_ids'][:j]
                target_id = sentence['input_ids'][j]

                input_target_pairs.append({
                    'input_tokens': ' '.join(input_tokens),
                    'target_token': target_token,
                    'input_ids': input_ids,
                    'target_id': target_id
                })

        json_file = os.path.join(output_dir, f'{prefix}_chunk_{i}.json')
        with open(json_file, 'w', encoding='utf-8') as f:  # Ensure UTF-8 encoding
            json.dump(input_target_pairs, f, ensure_ascii=False)  # Ensure proper encoding of non-ASCII characters

        logging.info(f"Saved chunk {i+1}/{num_chunks} to {json_file}")



# Main function to prepare shards
def prepare_shards(rank, world_size, output_dir):
    # Load and split the dataset
    train_data, valid_data, test_data = load_and_split_dataset()

    # Save training data per node
    train_file = save_data_per_node(train_data, rank, world_size, output_dir)

    # Load the custom tokenizer
    tokenizer_path = os.path.join(os.path.dirname(__file__), 'telugu_tokenizer_50k.json')
    tokenizer = Tokenizer.from_file(tokenizer_path)  # Ensure this path is correct

    # Tokenize and filter the sentences
    tokenized_train = tokenize_and_filter_sentences(train_file, tokenizer)

    # Create input-target pairs and save as JSON files for training
    create_input_target_pairs(tokenized_train, os.path.join(output_dir, 'train'), prefix=f'rank_{rank}_train')

    # Only rank 0 handles validation and test data preparation
    if rank == 0:
        # Tokenize and filter validation and test sets
        tokenized_valid = tokenize_and_filter_sentences(valid_data.to_json(), tokenizer)
        tokenized_test = tokenize_and_filter_sentences(test_data.to_json(), tokenizer)

        # Save validation and test data
        create_input_target_pairs(tokenized_valid, os.path.join(output_dir, 'valid'), prefix='valid')
        create_input_target_pairs(tokenized_test, os.path.join(output_dir, 'test'), prefix='test')

        logging.info("Saved validation and test datasets on rank 0")


# Run the shard preparation
if __name__ == "__main__":
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    output_dir = "shards"
    
    prepare_shards(rank, world_size, output_dir)
