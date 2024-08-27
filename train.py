import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import GPT2LMHeadModel, GPT2Config, get_scheduler
from datasets import load_dataset, DatasetDict, load_from_disk
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import GradScaler, autocast
from tqdm.auto import tqdm
import logging
import time
from datetime import datetime
import hashlib
import shutil
from torch.utils.tensorboard import SummaryWriter

# Set up logging
log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f"train_telugu_gen_{timestamp}.log")

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', handlers=[
    logging.FileHandler(log_file),
    logging.StreamHandler(sys.stdout)
])

# Set NCCL debugging level
os.environ['NCCL_DEBUG'] = 'INFO'

# --- Setup Function ---
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '172.26.115.220'
    os.environ['MASTER_PORT'] = '29500'
    logging.info(f"[Rank {rank}] Setting up the distributed environment: MASTER_ADDR={os.environ['MASTER_ADDR']}, MASTER_PORT={os.environ['MASTER_PORT']}, World Size={world_size}")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(0)
    logging.info(f"[Rank {rank}] Distributed environment setup complete.")
    dist.barrier()

# --- Cleanup Function ---
def cleanup(rank):
    logging.info(f"[Rank {rank}] Cleaning up the distributed environment.")
    dist.destroy_process_group()

# --- Custom Tokenizer Loading ---
def load_custom_tokenizer(tokenizer_path):
    from transformers import PreTrainedTokenizerFast
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    tokenizer.model_max_length = 256
    return tokenizer

# --- Dynamic Padding Collate Function ---
def dynamic_padding_collate_fn(batch, tokenizer):
    max_length = 256  # Fixed max length
    padded_input_ids = torch.full((len(batch), max_length), tokenizer.pad_token_id, dtype=torch.long)
    padded_labels = torch.full((len(batch), max_length), -100, dtype=torch.long)  # Use -100 for ignored index

    for i, example in enumerate(batch):
        input_ids = example['input_ids']
        labels = example['labels']
        seq_length = min(len(input_ids), max_length)

        padded_input_ids[i, :seq_length] = torch.tensor(input_ids[:seq_length], dtype=torch.long)
        padded_labels[i, :seq_length] = torch.tensor(labels[:seq_length], dtype=torch.long)

    attention_mask = (padded_input_ids != tokenizer.pad_token_id).int()

    return {
        'input_ids': padded_input_ids,
        'attention_mask': attention_mask,
        'labels': padded_labels
    }

# --- Hash Calculation Function ---
def calculate_hash(filepath, hash_func=hashlib.md5):
    """Calculates the hash of the file to check for integrity."""
    hash_obj = hash_func()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_obj.update(chunk)
    return hash_obj.hexdigest()

def calculate_directory_hash(dir_path, hash_func=hashlib.md5):
    """Calculates the hash of all files in a directory to check for integrity."""
    if not os.path.isdir(dir_path):
        logging.error(f"Directory does not exist: {dir_path}")
        return None
    
    hash_obj = hash_func()
    for root, dirs, files in os.walk(dir_path):
        for file in sorted(files):
            file_path = os.path.join(root, file)
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_obj.update(chunk)
    return hash_obj.hexdigest()

# --- Download and Verify Dataset ---
def download_and_verify_dataset(dataset_name, local_dir, token):
    """Downloads the dataset if not present or verifies its integrity if present."""
    dataset_path = os.path.join(local_dir, dataset_name.replace('/', '_'))
    dataset_cache = os.path.join(local_dir, "datasets")
    if not os.path.exists(dataset_cache):
        os.makedirs(dataset_cache)

    if os.path.exists(dataset_path):
        logging.info(f"Dataset found locally. Verifying integrity.")
        local_hash = calculate_directory_hash(dataset_path)
        if local_hash is None:
            logging.error(f"Failed to calculate local hash for dataset at {dataset_path}")
            return None

        remote_dataset = load_dataset(dataset_name, cache_dir=dataset_cache, token=token)
        temp_dataset_path = os.path.join(local_dir, "temp_dataset")
        remote_dataset.save_to_disk(temp_dataset_path)
        remote_hash = calculate_directory_hash(temp_dataset_path)
        if remote_hash is None:
            logging.error(f"Failed to calculate remote hash for dataset at {temp_dataset_path}")
            shutil.rmtree(temp_dataset_path)
            return None
        
        shutil.rmtree(temp_dataset_path)
        if local_hash != remote_hash:
            logging.info(f"Dataset integrity mismatch. Re-downloading dataset.")
            shutil.rmtree(dataset_path)
            dataset = load_dataset(dataset_name, cache_dir=dataset_cache, token=token)
            dataset.save_to_disk(dataset_path)
            logging.info(f"Dataset downloaded successfully.")
        else:
            logging.info(f"Dataset integrity verified.")
            dataset = DatasetDict.load_from_disk(dataset_path)
    else:
        logging.info(f"Dataset not found locally. Downloading to {dataset_path}.")
        dataset = load_dataset(dataset_name, cache_dir=dataset_cache, token=token)
        dataset.save_to_disk(dataset_path)
        logging.info(f"Dataset downloaded successfully.")
    return dataset

# --- Load Shard Function ---
def load_shard(rank, world_size, local_data_dir, dataset_name):
    dataset_path = os.path.join(local_data_dir, dataset_name.replace('/', '_'))
    logging.info(f"[Rank {rank}] Loading dataset from {dataset_path}.")
    dataset_dict = DatasetDict.load_from_disk(dataset_path)
    
    logging.info(f"[Rank {rank}] Sharding dataset.")
    shard_dict = {}
    for split, dataset in dataset_dict.items():
        shard_dict[split] = dataset.shard(num_shards=world_size, index=rank)
    
    shard_path = os.path.join(local_data_dir, f'shard_{rank}')
    DatasetDict(shard_dict).save_to_disk(shard_path)
    logging.info(f"[Rank {rank}] Shard saved to {shard_path}.")
    
    shard = DatasetDict.load_from_disk(shard_path)
    logging.info(f"[Rank {rank}] Shard sizes: { {split: len(shard[split]) for split in shard} }")
    
    return shard

# --- Training Function ---
def train(rank, world_size):
    setup(rank, world_size)

    # Hugging Face token
    hf_token = "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"
    local_data_dir = "./data"
    dataset_name = "Vipplav/telugu_pairs_1m"
    tokenizer_path = "./custom_tokenizer"

    # Download and verify dataset
    logging.info(f"[Rank {rank}] Checking dataset integrity and downloading if necessary.")
    dataset = download_and_verify_dataset(dataset_name, local_data_dir, hf_token)
    if dataset is None:
        logging.error(f"[Rank {rank}] Failed to load the dataset.")
        return

    dist.barrier()

    # Load shard
    shard = load_shard(rank, world_size, local_data_dir, dataset_name)

    # Initialize the custom tokenizer
    logging.info(f"[Rank {rank}] Initializing custom tokenizer.")
    tokenizer = load_custom_tokenizer(tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token

    # Create Distributed Sampler
    train_sampler = DistributedSampler(shard['train'], num_replicas=world_size, rank=rank)
    valid_sampler = DistributedSampler(shard['validation'], num_replicas=world_size, rank=rank)

    # Create DataLoader with the custom collate function
    logging.info(f"[Rank {rank}] Creating DataLoaders.")
    train_loader = DataLoader(shard['train'], batch_size=32, collate_fn=lambda x: dynamic_padding_collate_fn(x, tokenizer), sampler=train_sampler, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(shard['validation'], batch_size=32, collate_fn=lambda x: dynamic_padding_collate_fn(x, tokenizer), sampler=valid_sampler, num_workers=4, pin_memory=True)

    # Setup training
    logging.info(f"[Rank {rank}] Setting up model, optimizer, and scheduler.")
    config = GPT2Config(
        vocab_size=50000,
        n_embd=768,
        n_layer=8,
        n_head=8,
        pad_token_id=tokenizer.pad_token_id
    )
    model = GPT2LMHeadModel(config).cuda(0)
    model = DDP(model, device_ids=[0])

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    scaler = GradScaler()

    epochs = 3
    accumulation_steps = 8  # To maintain effective batch size
    max_grad_norm = 1.0

    num_training_steps = epochs * len(train_loader)
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=num_training_steps
    )

    # Setup TensorBoard
    if rank == 0:
        writer = SummaryWriter(log_dir=log_dir)

    # Training loop with tqdm progress bar, AMP, and TensorBoard
    logging.info(f"[Rank {rank}] Starting training.")
    model.train()
    best_loss = float('inf')
    best_model_path = './best_model'
    for epoch in range(epochs):
        total_loss = 0
        optimizer.zero_grad()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

        start_time = time.time()
        for batch_idx, batch in enumerate(progress_bar):
            batch = {k: v.cuda(0) for k, v in batch.items()}

            with autocast():
                outputs = model(**batch)
                loss = outputs.loss / accumulation_steps

            scaler.scale(loss).backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            progress_bar.set_postfix(loss=loss.item())
            total_loss += loss.item()

            if (batch_idx + 1) % 100 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                elapsed_time = time.time() - start_time
                logging.info(f"[Rank {rank}] Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {avg_loss:.4f}, Time elapsed: {elapsed_time:.2f} seconds")
                if rank == 0:
                    writer.add_scalar("Loss/train", avg_loss, epoch * len(train_loader) + batch_idx)

        avg_loss = total_loss / len(train_loader)
        logging.info(f"[Rank {rank}] Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")

        # Calculate and log perplexity
        perplexity = torch.exp(torch.tensor(avg_loss))
        logging.info(f"[Rank {rank}] Epoch {epoch+1} - Perplexity: {perplexity:.4f}")
        if rank == 0:
            writer.add_scalar("Perplexity/train", perplexity, epoch)

        if avg_loss < best_loss:
            best_loss = avg_loss
            if rank == 0:
                model.module.save_pretrained(best_model_path)
                tokenizer.save_pretrained(best_model_path)
                logging.info("[Rank 0] Saved new best model")

    logging.info(f"[Rank {rank}] Training completed.")
    if rank == 0:
        writer.close()

    if rank == 0:
        logging.info("[Rank 0] Uploading the best model to Hugging Face.")
        model.module.push_to_hub("my_finetuned_telugu_gpt2_model", token=hf_token)
        tokenizer.push_to_hub("my_finetuned_telugu_gpt2_model", token=hf_token)
        logging.info("[Rank 0] Best model uploaded successfully to Hugging Face.")

    dist.barrier()
    cleanup(rank)

# --- Start Training Function ---
def start_training():
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    logging.info(f"RANK: {rank}, LOCAL_RANK: {os.environ.get('LOCAL_RANK')}, WORLD_SIZE: {world_size}")
    try:
        train(rank, world_size)
    except Exception as e:
        logging.error(f"Error in training: {e}")
        sys.exit(1)

if __name__ == "__main__":
    start_training()
