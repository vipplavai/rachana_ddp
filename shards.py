import os
import json
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

    train_data = dataset.select(range(0, train_size))
    valid_data = dataset.select(range(train_size, train_size + valid_size))
    test_data = dataset.select(range(train_size + valid_size, total_sentences))

    return train_data, valid_data, test_data

# Save data for each node
def save_data_per_node(train_data, rank, world_size, output_dir):
    train_dir = os.path.join(output_dir, 'train')
    os.makedirs(train_dir, exist_ok=True)  # Ensure the train directory exists
    logging.info(f"Train directory for rank {rank}: {train_dir}")

    total_sentences = len(train_data)
    sentences_per_node = total_sentences // world_size

    start_idx = rank * sentences_per_node
    end_idx = start_idx + sentences_per_node if rank != world_size - 1 else total_sentences

    node_data = train_data.select(range(start_idx, end_idx))
    node_file = os.path.join(train_dir, f'node_{rank}.json')

    if len(node_data) == 0:
        logging.warning(f"No data assigned to rank {rank}. This can result in empty shard files.")
    else:
        sentences = [{"Sentence": item['Sentence']} for item in node_data]

        with open(node_file, 'w', encoding='utf-8') as f:
            json.dump(sentences, f, ensure_ascii=False, indent=4)  # Use indent for readability

        logging.info(f"Saved {len(sentences)} sentences to {node_file} for rank {rank}")
    
    return node_file

# Tokenize and filter sentences
def tokenize_and_filter_sentences(file_path, tokenizer):
    tokenized_sentences = []
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for entry in data:
            sentence = entry.get('Sentence', '')  # Use 'Sentence' as the key
            tokens = tokenizer.encode(sentence)
            if len(tokens.ids) <= 256:
                tokenized_sentences.append({
                    'input_tokens': tokens.tokens,
                    'input_ids': tokens.ids
                })

    logging.info(f"Tokenized and filtered {len(tokenized_sentences)} sentences from {file_path}")
    return tokenized_sentences

# Create input-target pairs and save as JSON files
def create_input_target_pairs(tokenized_sentences, output_dir, prefix, chunk_size=3000):
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Output directory for {prefix}: {output_dir}")

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
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(input_target_pairs, f, ensure_ascii=False)

        logging.info(f"Saved chunk {i+1}/{num_chunks} to {json_file}")

# Main function to prepare shards
def prepare_shards(rank, world_size, output_dir):
    # Load and split the dataset
    train_data, valid_data, test_data = load_and_split_dataset()

    # Save training data per node
    train_file = save_data_per_node(train_data, rank, world_size, output_dir)

    # Load the custom tokenizer
    tokenizer_path = os.path.join(os.path.dirname(__file__), 'telugu_tokenizer_50k.json')
    tokenizer = Tokenizer.from_file(tokenizer_path)

    # Tokenize and filter the sentences
    tokenized_train = tokenize_and_filter_sentences(train_file, tokenizer)

    # Create input-target pairs and save as JSON files for training
    create_input_target_pairs(tokenized_train, os.path.join(output_dir, 'train'), prefix=f'rank_{rank}_train')

    # Only rank 0 handles validation and test data preparation
    if rank == 0:
        valid_dir = os.path.join(output_dir, 'valid')
        test_dir = os.path.join(output_dir, 'test')
        os.makedirs(valid_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        logging.info(f"Validation directory: {valid_dir}")
        logging.info(f"Test directory: {test_dir}")

        # Tokenize and filter validation and test sets
        valid_data_file = os.path.join(valid_dir, 'valid.json')
        test_data_file = os.path.join(test_dir, 'test.json')
        
        # Save valid and test data without using the encoding parameter
        valid_data.to_json(valid_data_file, force_ascii=False)
        test_data.to_json(test_data_file, force_ascii=False)

        # Process and save the tokenized data
        tokenized_valid = tokenize_and_filter_sentences(valid_data_file, tokenizer)
        tokenized_test = tokenize_and_filter_sentences(test_data_file, tokenizer)

        # Save validation and test data
        create_input_target_pairs(tokenized_valid, valid_dir, prefix='valid')
        create_input_target_pairs(tokenized_test, test_dir, prefix='test')

        logging.info("Saved validation and test datasets on rank 0")


# Run the shard preparation
if __name__ == "__main__":
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    output_dir = os.path.join(os.path.dirname(__file__), 'shards')
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Output base directory: {output_dir}")
    
    prepare_shards(rank, world_size, output_dir)
