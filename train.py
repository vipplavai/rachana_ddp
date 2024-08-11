import os
import json
import torch
import torch.distributed as dist
import logging
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.utils.rnn import pad_sequence
from transformers import GPT2LMHeadModel, GPT2Config
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import random
import math
from shards import prepare_shards  # Import the prepare_shards function from shards.py

# Set NCCL debug level to info
os.environ['NCCL_DEBUG'] = 'INFO'

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Setup Function ---
def setup(rank, world_size):
    torch.cuda.empty_cache()
    os.environ['MASTER_ADDR'] = '172.26.115.220'  # Set this to your master node's address
    os.environ['MASTER_PORT'] = '29500'
    logging.info(f"[Rank {rank}] Setting up the distributed environment: MASTER_ADDR={os.environ['MASTER_ADDR']}, MASTER_PORT={os.environ['MASTER_PORT']}, World Size={world_size}")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(0)  # Set CUDA device to 0
    logging.info(f"[Rank {rank}] Distributed environment setup complete.")

    # Print GPU information
    total_gpus = dist.get_world_size()
    available_memory_per_gpu = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # in GB
    logging.info(f"[Rank {rank}] Total GPUs: {total_gpus}, Available Memory per GPU: {available_memory_per_gpu:.2f} GB")

    # Collect total available memory across all GPUs
    collective_memory = available_memory_per_gpu * total_gpus
    logging.info(f"[Rank {rank}] Collective GPU Memory: {collective_memory:.2f} GB")

    dist.barrier()  # Ensure all processes have finished initialization

# --- Cleanup Function ---
def cleanup(rank):
    logging.info(f"[Rank {rank}] Cleaning up the distributed environment.")
    dist.destroy_process_group()

# --- Helper Functions ---
def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt

# Paths and settings
train_dir = "shards/train"
valid_dir = "shards/valid"
test_dir = "shards/test"
output_dir = "rachana_ddp/shards"
os.makedirs(output_dir, exist_ok=True)

# Initialize the GPT-2 model
config = GPT2Config(
    vocab_size=50000,
    n_positions=1024,
    n_ctx=1024,
    n_embd=768,
    n_layer=8,
    n_head=8,
    resid_pdrop=0.1,
    attn_pdrop=0.1,
)
model = GPT2LMHeadModel(config)
model.to('cuda:0')


# Dataset class
class TextDataset(Dataset):
    def __init__(self, input_ids, target_ids):
        self.input_ids = input_ids
        self.target_ids = target_ids

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

# Collate function with dynamic padding
def collate_fn(batch):
    inputs, targets = zip(*batch)
    inputs = [torch.tensor(seq) for seq in inputs]
    targets = torch.tensor(targets)
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0)
    return inputs_padded, targets

# Load JSON data
def load_data_from_shard(shard_file):
    with open(shard_file, 'r') as f:
        data = [json.loads(line) for line in f]
    input_ids = [entry['input_ids'] for entry in data]
    target_ids = [entry['target_id'] for entry in data]
    return input_ids, target_ids

# Calculate perplexity
def calculate_perplexity(loss):
    return math.exp(loss)

# Training loop with file-by-file processing
def train(model, checkpoint_interval=5000, print_loss_every=300):
    global global_step, total_steps, total_tokens
    model.train()

    best_checkpoints = []

    while global_step < checkpoint_interval:
        total_loss = 0
        total_train_tokens = 0

        # Shuffle the shard files for each run
        shard_files = os.listdir(train_dir)
        random.shuffle(shard_files)

        for shard_file in shard_files:
            train_shard_path = os.path.join(train_dir, shard_file)
            input_ids, target_ids = load_data_from_shard(train_shard_path)
            train_dataset = TextDataset(input_ids, target_ids)
            batch_size = 8
            
            # Use DistributedSampler to ensure each process gets a different subset of data
            train_sampler = DistributedSampler(train_dataset)
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, sampler=train_sampler)
            
            num_batches = len(train_dataloader)
            logging.info(f"\nProcessing file: {shard_file}")
            logging.info(f"Number of input-target pairs in this file: {len(train_dataset)}")
            logging.info(f"Number of batches in this file: {num_batches}")

            # Progress bar for batches within this shard
            batch_progress = tqdm(train_dataloader, desc=f"Training on {shard_file}", leave=True, dynamic_ncols=True)
            for step, (inputs, targets) in enumerate(batch_progress):
                inputs, targets = inputs.to('cuda:0'), targets.to('cuda:0')
                
                with autocast():
                    outputs = model(inputs, labels=inputs)
                    loss = outputs.loss / gradient_accumulation_steps
                    total_loss += loss.item()

                scaler.scale(loss).backward()

                total_train_tokens += inputs.numel()  # Count the number of tokens in the batch

                if (step + 1) % gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1
                    total_steps += 1

                    # Aggregate and log loss, perplexity, token count, and learning rate to TensorBoard (only for rank 0)
                    if dist.get_rank() == 0:
                        avg_loss = reduce_tensor(torch.tensor(total_loss / num_batches).to('cuda:0'), dist.get_world_size()).item()
                        avg_perplexity = calculate_perplexity(avg_loss)
                        total_tokens_reduced = reduce_tensor(torch.tensor(total_train_tokens).to('cuda:0'), dist.get_world_size()).item()
                        lr = scheduler.get_last_lr()[0]

                        writer.add_scalar("Loss/train", avg_loss, global_step)
                        writer.add_scalar("Perplexity/train", avg_perplexity, global_step)
                        writer.add_scalar("Tokens/train", total_tokens_reduced, global_step)
                        writer.add_scalar("Learning Rate", lr, global_step)

                    # Print stats every `print_loss_every` steps
                    if global_step % print_loss_every == 0:
                        logging.info(f"Step {global_step}: Loss = {avg_loss}, Perplexity = {avg_perplexity}, Tokens Processed = {total_tokens_reduced}")

                    # Every checkpoint_interval steps, evaluate and save
                    if global_step % checkpoint_interval == 0:
                        avg_loss = total_loss / num_batches
                        evaluate(model, global_step, total_train_tokens)
                        if dist.get_rank() == 0:
                            save_checkpoint_and_embeddings(model, global_step, avg_loss, best_checkpoints)
                        return  # Stop training after the interval is reached

def evaluate(model, global_step, total_train_tokens):
    model.eval()
    total_eval_loss = 0
    total_eval_tokens = 0

    with torch.no_grad():
        for shard_file in tqdm(os.listdir(valid_dir), desc="Evaluating", colour='green'):
            valid_shard_path = os.path.join(valid_dir, shard_file)
            input_ids, target_ids = load_data_from_shard(valid_shard_path)
            valid_dataset = TextDataset(input_ids, target_ids)
            valid_sampler = DistributedSampler(valid_dataset, shuffle=False)
            valid_dataloader = DataLoader(valid_dataset, batch_size=8, collate_fn=collate_fn, sampler=valid_sampler)
            
            for inputs, targets in valid_dataloader:
                inputs, targets = inputs.to('cuda:0'), targets.to('cuda:0')
                outputs = model(inputs, labels=inputs)
                loss = outputs.loss
                total_eval_loss += loss.item()
                total_eval_tokens += inputs.numel()

    avg_eval_loss = total_eval_loss / len(os.listdir(valid_dir))
    avg_perplexity = calculate_perplexity(avg_eval_loss)
    total_eval_tokens_reduced = reduce_tensor(torch.tensor(total_eval_tokens).to('cuda:0'), dist.get_world_size()).item()

    logging.info(f"Validation Loss: {avg_eval_loss}, Perplexity: {avg_perplexity}")
    if dist.get_rank() == 0:
        writer.add_scalar("Loss/valid", avg_eval_loss, global_step)
        writer.add_scalar("Perplexity/valid", avg_perplexity, global_step)
        writer.add_scalar("Tokens/valid", total_eval_tokens_reduced, global_step)
    model.train()

def save_checkpoint_and_embeddings(model, step, average_loss, best_checkpoints):
    if dist.get_rank() == 0:
        checkpoint_path = os.path.join(output_dir, f"step_{step}.pt")
        embedding_path = os.path.join(output_dir, f"step_{step}_embeddings.pt")
        
        # Save model checkpoint
        torch.save(model.state_dict(), checkpoint_path)
        
        # Save model embeddings
        torch.save(model.module.transformer.wte.weight.data, embedding_path)
        
        logging.info(f"Checkpoint and embeddings saved for step {step}.")

        # Maintain only the best 2 checkpoints
        best_checkpoints.append((checkpoint_path, embedding_path, average_loss))
        best_checkpoints.sort(key=lambda x: x[2])  # Sort by loss, lowest first
        
        if len(best_checkpoints) > 2:
            # Remove the worst checkpoint
            worst_checkpoint, worst_embedding, _ = best_checkpoints.pop()
            os.remove(worst_checkpoint)
            os.remove(worst_embedding)
            logging.info(f"Removed old checkpoint and embeddings: {worst_checkpoint}, {worst_embedding}")

# --- Main execution ---
if __name__ == "__main__":
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    # Setup DDP environment
    setup(rank, world_size)

    # Wrap the model with DDP after initializing the process group
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[0], output_device=0, find_unused_parameters=True)

    # Mixed precision and gradient accumulation setup
    scaler = GradScaler()
    gradient_accumulation_steps = 4  # Accumulate gradients over 4 batches
    global_step = 0
    total_steps = 0  # We'll track the total steps
    total_tokens = 0  # Track the total number of tokens processed

    # Optimizer and scheduler setup
    optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=500, num_training_steps=100000)  # Just a placeholder value

    # TensorBoard setup (only for rank 0)
    if dist.get_rank() == 0:
        writer = SummaryWriter(log_dir=os.path.join(output_dir, "logs"))

    try:
        # Prepare the shards
        prepare_shards(rank, world_size, output_dir)

        # Start training
        train(model, checkpoint_interval=5000, print_loss_every=300)

        # Evaluation should be performed only by the master node (rank 0)
        if dist.get_rank() == 0:
            evaluate(model, global_step=global_step, total_train_tokens=total_tokens)
    finally:
        if dist.get_rank() == 0:
            writer.close()
        cleanup(rank)
