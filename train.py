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
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
import subprocess
import webbrowser
from tokenizers import Tokenizer
from time import time, sleep
import math
from dotenv import load_dotenv
import csv
from huggingface_hub import HfApi


# Load environment variables from .env file
load_dotenv()

# Set NCCL environment variables to avoid common issues
# Set NCCL environment variables to avoid common issues
os.environ['NCCL_DEBUG'] = 'INFO'
# os.environ['NCCL_DEBUG_SUBSYS'] = 'ALL'
os.environ['NCCL_PROTO'] = 'Simple'
os.environ['NCCL_ALGO'] = 'Ring'
os.environ['NCCL_IB_DISABLE'] = '1'  # Disable InfiniBand if not used
os.environ['NCCL_BLOCKING_WAIT'] = '1'
os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'
os.environ['NCCL_TIMEOUT'] = '1800'  # Set a longer timeout in seconds
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'


# Set CUDA_VISIBLE_DEVICES early in the script
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Ensure only GPU 0 is used

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to check and kill busy CUDA devices
def check_and_kill_busy_cuda_device():
    try:
        # Check if any process is using the CUDA device
        result = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_bus_id', '--format=csv,noheader']).decode('utf-8').strip()
        if result:
            # Extract the PID and kill the process
            lines = result.splitlines()
            for line in lines:
                pid, _ = line.split(',')
                pid = pid.strip()
                if pid:
                    subprocess.call(['kill', '-9', pid])
                    logging.info(f"Killed process with PID {pid} that was using the CUDA device.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to check CUDA devices: {e}")

# Call this function before setting up DDP
check_and_kill_busy_cuda_device()

# --- Setup Function ---
def setup(rank, world_size):
    torch.cuda.empty_cache()
    os.environ['MASTER_ADDR'] = '172.26.115.220'  # Set this to your master node's address
    os.environ['MASTER_PORT'] = '29500'
    logging.info(f"[Rank {rank}] Setting up the distributed environment: MASTER_ADDR={os.environ['MASTER_ADDR']}, MASTER_PORT={os.environ['MASTER_PORT']}, World Size={world_size}")
    
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(0)  # Always use device 0
    
    # Barrier to ensure all processes have initialized properly
    logging.info(f"[Rank {rank}] Reached barrier after setup.")
    try:
        dist.barrier()
        logging.info(f"[Rank {rank}] Passed barrier after setup.")
    except Exception as e:
        logging.error(f"Error during barrier synchronization after setup: {e}")
        raise

    logging.info(f"[Rank {rank}] Distributed environment setup complete.")
    logging.info(f"Using device: {torch.cuda.current_device()} with device name: {torch.cuda.get_device_name(0)}")


# --- Cleanup Function ---
def cleanup(rank):
    logging.info(f"[Rank {rank}] Cleaning up the distributed environment.")
    dist.destroy_process_group()

# --- Checkpoint Saving Function ---
def save_checkpoint_and_embeddings(model, epoch, avg_loss, best_checkpoints):
    if dist.get_rank() == 0:
        checkpoint_path = os.path.join(output_dir, f"epoch_{epoch}.pt")
        embedding_path = os.path.join(output_dir, f"epoch_{epoch}_embeddings.pt")
        
        torch.save(model.state_dict(), checkpoint_path)
        torch.save(model.module.transformer.wte.weight.data, embedding_path)
        
        logging.info(f"Checkpoint and embeddings saved for epoch {epoch}.")

        best_checkpoints.append((checkpoint_path, embedding_path, avg_loss))
        best_checkpoints.sort(key=lambda x: x[2])  # Sort by loss, lowest first
        
        if len(best_checkpoints) > 2:
            worst_checkpoint, worst_embedding, _ = best_checkpoints.pop()
            os.remove(worst_checkpoint)
            os.remove(worst_embedding)
            logging.info(f"Removed old checkpoint and embeddings: {worst_checkpoint}, {worst_embedding}")

# Paths and settings
node_output_dir = os.path.dirname(__file__)  # Directory where train.py is located
output_dir = os.path.join(node_output_dir, 'outputs')  # Output directory
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
assert torch.cuda.is_available(), "CUDA is not available. Please check your CUDA installation."
model.to('cuda:0')

# Get the directory of the current script (train.py)
script_dir = os.path.dirname(os.path.realpath(__file__))
tokenizer_path = os.path.join(script_dir, 'telugu_tokenizer_50k.json')
tokenizer = Tokenizer.from_file(tokenizer_path)

# Function to load Hugging Face dataset and split
def load_and_split_dataset():
    dataset = load_dataset('KPrashanth/Telugu_sentences', split='train')
    count_sentences = len(dataset)
    sample_size = int(0.00001 * count_sentences)  # Take 1% of the data
    dataset = dataset.shuffle(seed=42).select(range(sample_size))
    total_sentences = len(dataset)
    logging.info(f"Total sentences in sampled dataset: {total_sentences}")
    return dataset

# Function to create input-target pairs
def create_input_target_pairs(tokenized_sentences):
    input_target_pairs = []
    for tokens in tokenized_sentences:
        for i in range(1, len(tokens)):
            input_seq = tokens[:i]
            target_token = tokens[i]
            input_target_pairs.append({"input_ids": input_seq, "target_id": target_token})
    return input_target_pairs

# Function to save input-target pairs in JSON format
def save_input_target_pairs(input_target_pairs, output_file):
    with open(output_file, 'a') as f:
        for pair in input_target_pairs:
            f.write(json.dumps(pair) + '\n')

def process_dataset(dataset, rank, world_size, batch_size=3000):
    # Set seed for consistent sharding across nodes
    seed = 42
    torch.manual_seed(seed)
    dataset = dataset.shuffle(seed=seed)  # Shuffle with seed for consistent sharding
    dataset = split_dataset_by_node(dataset, rank=rank, world_size=world_size)
    logging.info(f"Node {rank}: First sentence: {dataset[0]['Sentence']}")
    tokenized_sentences = [tokenizer.encode(d['Sentence']).ids for d in dataset]
    logging.info(f"Node {rank}: Total number of sentences: {len(dataset)}")
    
    output_file = os.path.join(output_dir, f"input_target_pairs_{rank}.json")
    if os.path.exists(output_file):
        os.remove(output_file)
        logging.info(f"Existing file {output_file} deleted.")
    
    input_target_pairs = []
    for i in range(0, len(tokenized_sentences), batch_size):
        batch = tokenized_sentences[i:i+batch_size]
        input_target_pairs_batch = create_input_target_pairs(batch)
        save_input_target_pairs(input_target_pairs_batch, output_file)
        logging.info(f"Node {rank}: Processed batch {i//batch_size + 1}")

    return output_file


def load_data_from_shard(shard_file):
    input_ids = []
    target_ids = []
    with open(shard_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            input_ids.append(data['input_ids'])
            target_ids.append(data['target_id'])
    return input_ids, target_ids

class TextDataset(torch.utils.data.Dataset):
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

# Function to reduce and average metrics across all nodes
def reduce_and_average_metrics(metrics, world_size):
    reduced_metrics = {key: value.clone().detach().cuda().float() if isinstance(value, torch.Tensor) else torch.tensor(value).cuda().float() for key, value in metrics.items()}
    
    torch.cuda.synchronize()  # Ensure all CUDA operations are complete

    for key in reduced_metrics:
        try:
            dist.all_reduce(reduced_metrics[key], op=dist.ReduceOp.SUM)
        except Exception as e:
            logging.error(f"Error during NCCL all_reduce operation: {e}")
            raise

        reduced_metrics[key] /= world_size

    return {key: value.item() for key, value in reduced_metrics.items()}
 

# CSV Logging Function
def log_metrics_to_csv(csv_file, metrics):
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=metrics.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics)

# Function to evaluate model
def evaluate(model, epoch, dataloader):
    model.module.eval()  # Access the underlying model for evaluation
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to('cuda:0'), targets.to('cuda:0')
            with autocast():
                outputs = model.module(inputs, labels=inputs)  # Use model.module
                loss = outputs.loss
                total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    logging.info(f"Epoch {epoch}: Evaluation loss = {avg_loss}")
    return avg_loss


import os
import time
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# Training loop with epoch-based processing
def train(model, num_epochs, print_loss_every=300):  # Pass num_epochs as a parameter
    global total_steps, total_tokens
    model.train()

    best_checkpoints = []
    csv_file = os.path.join(output_dir, "metrics.csv")

    dataset = load_and_split_dataset()
    shard_file = process_dataset(dataset, rank, world_size)
    input_ids, target_ids = load_data_from_shard(shard_file)
    train_dataset = TextDataset(input_ids, target_ids)
    batch_size = 8
    
    train_sampler = DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, sampler=train_sampler)
    
    for epoch in range(1, num_epochs + 1):  # Use num_epochs here
        total_loss = 0
        total_train_tokens = 0
        start_time = time.time()

        # Log the start of training for this epoch for each rank
        logging.info(f"[Rank {rank}] Epoch {epoch} training started.")

        if dist.get_rank() == 0:
            batch_progress = tqdm(train_dataloader, desc=f"[Rank {rank}] Training on {shard_file}", leave=True, dynamic_ncols=True)
        else:
            batch_progress = tqdm(train_dataloader, desc=f"[Rank {rank}] Training", leave=True, dynamic_ncols=True)

        for step, (inputs, targets) in enumerate(batch_progress):
            inputs, targets = inputs.to('cuda:0'), targets.to('cuda:0')
            
            with autocast():
                outputs = model(inputs, labels=inputs)
                loss = outputs.loss / gradient_accumulation_steps
                total_loss += loss.item()

            scaler.scale(loss).backward()

            total_train_tokens += inputs.numel()

            if (step + 1) % gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                total_steps += 1

                if (step + 1) % print_loss_every == 0:
                    logging.info(f"[Rank {rank}] Step {step + 1}: Current Loss = {total_loss / (step + 1)}")

                local_metrics = {
                    "train_loss": total_loss / len(train_dataloader),
                    "train_perplexity": calculate_perplexity(total_loss / len(train_dataloader)),
                    "train_tokens": total_train_tokens,
                    "learning_rate": scheduler.get_last_lr()[0],
                    "comm_time": time.time() - start_time
                }

                global_metrics = reduce_and_average_metrics(local_metrics, dist.get_world_size())

                if dist.get_rank() == 0:
                    writer.add_scalar("Loss/train", global_metrics["train_loss"], total_steps)
                    writer.add_scalar("Perplexity/train", global_metrics["train_perplexity"], total_steps)
                    writer.add_scalar("Tokens/train", global_metrics["train_tokens"], total_steps)
                    writer.add_scalar("Learning Rate", global_metrics["learning_rate"], total_steps)
                    writer.add_scalar("Comm Time/train", global_metrics["comm_time"], total_steps)

                    log_metrics_to_csv(csv_file, {"epoch": epoch, "step": total_steps, **global_metrics})

                    if total_steps % print_loss_every == 0:
                        logging.info(f"Step {total_steps}: Loss = {global_metrics['train_loss']}, Perplexity = {global_metrics['train_perplexity']}, Tokens Processed = {global_metrics['train_tokens']}")

        # Log the completion of training for this epoch for each rank
        logging.info(f"[Rank {rank}] Completed epoch {epoch} with average loss: {total_loss / len(train_dataloader)}")

        # Barrier to ensure all nodes complete the epoch before proceeding
        logging.info(f"[Rank {rank}] Waiting at dist.barrier() before completing epoch {epoch}.")
        dist.barrier()
        logging.info(f"[Rank {rank}] Passed dist.barrier() after completing epoch {epoch}.")

        if dist.get_rank() == 0:
            # Log that the master node is waiting for worker nodes
            logging.info(f"[Rank 0] Waiting for all ranks to finish epoch {epoch} training.")
            
            avg_eval_loss = evaluate(model, epoch, train_dataloader)  # Use epoch from loop
            save_checkpoint_and_embeddings(model, epoch, avg_eval_loss, best_checkpoints)

            # Log confirmation that Rank 0 has received training data from all nodes
            logging.info(f"[Rank 0] Completed epoch {epoch} and saved model checkpoint after receiving data from all nodes.")

        # Barrier to ensure the model is saved only after all nodes have completed their training and Rank 0 has finished evaluation
        logging.info(f"[Rank {rank}] Waiting at final dist.barrier() after epoch {epoch} completion.")
        dist.barrier()
        logging.info(f"[Rank {rank}] Passed final dist.barrier() after epoch {epoch} completion.")



        
# Function to check if TensorBoard port is available
def is_port_available(port):
    try:
        subprocess.check_output(["lsof", "-i", f":{port}"])
        return False
    except subprocess.CalledProcessError:
        return True

# Function to kill the process occupying the TensorBoard port
def kill_tensorboard_process(port):
    try:
        result = subprocess.check_output(["lsof", "-t", f":{port}"]).decode().strip()
        if result:
            subprocess.call(['kill', '-9', result])
            logging.info(f"Killed process occupying port {port}.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to kill process on port {port}: {e}")

# Function to launch TensorBoard after the first event is written
def launch_tensorboard(log_dir):
    port = 6006
    if not is_port_available(port):
        kill_tensorboard_process(port)

    command = ["tensorboard", "--logdir", log_dir, "--host", "0.0.0.0", "--port", str(port)]
    subprocess.Popen(command)
    webbrowser.open(f"http://localhost:{port}", new=2)


def calculate_perplexity(loss):
    if isinstance(loss, float):
        loss = torch.tensor(loss)
    return torch.exp(loss)

# Function to push selected files to Hugging Face repository
def push_to_huggingface_repo():
    api = HfApi()
    repo_id = os.environ['HUGGINGFACE_REPO_ID']
    token = os.environ['HUGGINGFACE_TOKEN']
    
    # Initialize repository
    repo = Repository(local_dir=output_dir, clone_from=repo_id, use_auth_token=token)
    
    # List of files to include in the upload
    files_to_include = ["train.py", "auto.py", "metrics.csv", "epoch_1.pt", "epoch_1_embeddings.pt", "README.md", ".gitattributes"]

    # Upload the selected files
    for file in files_to_include:
        file_path = os.path.join(output_dir, file)
        if os.path.exists(file_path):
            repo.add(file_path)
        else:
            print(f"File {file_path} does not exist and will not be uploaded.")
    
    # Commit and push to Hugging Face Hub
    repo.push_to_hub(commit_message="Updated model and training files, excluding input-target pairs JSON files.")
    

# --- Main execution ---
if __name__ == "__main__":
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    num_epochs = 5  # Define the number of epochs here

    setup(rank, world_size)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[0], output_device=0, find_unused_parameters=False)
    scaler = GradScaler()
    gradient_accumulation_steps = 4
    total_steps = 0
    total_tokens = 0

    optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=500, num_training_steps=100000)

    if dist.get_rank() == 0:
        writer = SummaryWriter(log_dir=os.path.join(output_dir, "logs"))

    try:
        if dist.get_rank() == 0:
            launch_tensorboard(log_dir=os.path.join(output_dir, "logs"))
        
        # Barrier to ensure all nodes start training at the same time
        dist.barrier()
        
        # Pass the num_epochs variable to the train function
        train(model, num_epochs=num_epochs, print_loss_every=300)

        # Barrier before evaluation to ensure training has completed
        dist.barrier()

        if dist.get_rank() == 0:
            evaluate(model, epoch=num_epochs, dataloader=train_dataloader)  # Use num_epochs

        # Barrier before cleanup to ensure all evaluations are done
        dist.barrier()

    finally:
        if dist.get_rank() == 0:
            writer.close()
            push_to_huggingface_repo()
        cleanup(rank)
