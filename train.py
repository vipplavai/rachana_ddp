import os
import sys
import json
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, distributed
from torch.cuda.amp import autocast, GradScaler
from transformers import GPT2Config, GPT2Model, get_linear_schedule_with_warmup
from datasets import load_dataset, load_from_disk
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import subprocess
import logging
from datetime import datetime, timedelta
import hashlib
import time

# Setup logging configuration
log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f"train_telugu_gen_{timestamp}.log")

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s [%(process)d] %(message)s', handlers=[
    logging.FileHandler(log_file),
    logging.StreamHandler(sys.stdout)
])

# Set NCCL debugging and timeout levels
os.environ['NCCL_DEBUG'] = 'INFO'
os.environ['NCCL_SOCKET_TIMEOUT'] = '200'  # Increase timeout to avoid premature termination

# Function to kill existing CUDA processes except the current one
def kill_existing_cuda_processes():
    current_pid = os.getpid()
    try:
        output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'])
        pids = [int(pid) for pid in output.decode().splitlines() if pid.isdigit()]
        for pid in pids:
            if pid != current_pid:
                logging.info(f"Killing CUDA process with PID: {pid}")
                os.kill(pid, 9)
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to fetch CUDA processes: {e}")
    except Exception as e:
        logging.error(f"Failed to kill existing CUDA processes: {e}")

def parse_arguments():
    parser = argparse.ArgumentParser(description="DDP Training Script")
    parser.add_argument('--start_phase', type=int, required=True, help='Starting phase number')
    parser.add_argument('--end_phase', type=int, required=True, help='Ending phase number')
    return parser.parse_args()

def setup(rank, world_size):
    kill_existing_cuda_processes()  # Kill existing CUDA processes at the start

    # Set up environment variables for DDP
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', '172.26.112.16')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500')
    
    logging.info(f"[Rank {rank}] Setting up the distributed environment: MASTER_ADDR={os.environ['MASTER_ADDR']}, MASTER_PORT={os.environ['MASTER_PORT']}, World Size={world_size}")
    
    # Initialize process group
    try:
        dist.init_process_group(backend="nccl", init_method='env://', rank=rank, world_size=world_size)
        torch.cuda.set_device(0)  # Force the CUDA device ID to be 0
        logging.info(f"[Rank {rank}] Distributed environment setup complete on device 0.")
        dist.barrier()  # Synchronize all processes
    except Exception as e:
        logging.error(f"[Rank {rank}] Failed to initialize the distributed process group: {e}")
        sys.exit(1)

    # Print GPU power usage for this rank
    total_power = print_gpu_power(rank)
    dist.barrier()  # Synchronize after GPU power logging

    # Collectively aggregate the power usage across all ranks
    try:
        collective_power = torch.tensor([total_power], dtype=torch.float32).cuda(0)  # Use device 0
        dist.all_reduce(collective_power, op=dist.ReduceOp.SUM)
        if rank == 0:
            logging.info(f"[Rank 0] Collective GPU Power Usage: {collective_power.item()} W")
    except Exception as e:
        logging.error(f"[Rank {rank}] Error during power aggregation: {e}")

def print_gpu_power(rank):
    """
    Function to log the current GPU power usage.
    """
    try:
        output = subprocess.check_output(['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'])
        power_usage = float(output.decode().splitlines()[rank])  # Assumes rank maps directly to GPU index
        logging.info(f"[Rank {rank}] GPU Power Usage: {power_usage} W")
        return power_usage
    except (subprocess.CalledProcessError, IndexError) as e:
        logging.error(f"[Rank {rank}] Failed to query GPU power usage: {e}")
        return 0.0

def log_gpu_memory():
    """
    Log GPU memory statistics for the current rank (always device 0).
    """
    total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    reserved_memory = torch.cuda.memory_reserved(0) / (1024**3)
    allocated_memory = torch.cuda.memory_allocated(0) / (1024**3)
    free_memory = reserved_memory - allocated_memory
    logging.info(f"[Device 0] Total GPU Memory: {total_memory:.2f} GB, Allocated: {allocated_memory:.2f} GB, Free: {free_memory:.2f} GB")

def load_config(config_path):
    """
    Load configuration from a JSON file.
    """
    with open(config_path, "r") as file:
        config = json.load(file)
    return config


def print_phase_info(config, start_phase, end_phase):
    """
    Print phase information including dataset names and other details.
    """
    logging.info("\nLoaded Configuration Details:")
    logging.info(f"Model Configuration: {config['model_config']}")
    
    logging.info("\nTraining Control:")
    logging.info(f"Starting Phase: {start_phase}")
    logging.info(f"Ending Phase: {end_phase}")
    
    logging.info("\nPhase Details:")
    for phase_num in range(start_phase, end_phase + 1):
        phase_key = f"phase{phase_num}"
        phase_info = config["phases"].get(phase_key)
        if phase_info:
            logging.info(f"\nPhase {phase_num}:")
            logging.info(f"  - Dataset Name: {phase_info['dataset_name']}")
            logging.info(f"  - Checkpoint Directory: {phase_info['checkpoint_dir']}")
            logging.info(f"  - Log Directory: {phase_info['log_dir']}")
            logging.info(f"  - Batch Size: {phase_info['batch_size']}")
            logging.info(f"  - Learning Rate: {phase_info['learning_rate']}")
            logging.info(f"  - Epochs: {phase_info['epochs']}")
            logging.info(f"  - Gradient Accumulation Steps: {phase_info['gradient_accumulation_steps']}")
        else:
            logging.info(f"Phase {phase_num} configuration not found in the config file.")

def calculate_hash(file_path, chunk_size=1024*1024):
    """
    Calculate the MD5 hash of a file to verify integrity.
    """
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                hash_md5.update(chunk)
    except FileNotFoundError:
        return None
    return hash_md5.hexdigest()

def verify_dataset_integrity(dataset_dir):
    """
    Verify the integrity of a locally saved dataset by checking its files.
    """
    return os.path.isdir(dataset_dir) and any(f.endswith('.arrow') or f.endswith('.parquet') for f in os.listdir(dataset_dir))

def save_dataset_to_disk(dataset, save_path):
    """
    Save the dataset to disk and confirm with print statements.
    """
    os.makedirs(save_path, exist_ok=True)
    dataset.save_to_disk(save_path)
    logging.info(f"Dataset successfully saved to {save_path}")
    
    loaded_dataset = load_from_disk(save_path)
    logging.info(f"Loaded dataset from disk at {save_path} with {len(loaded_dataset)} records.")

def download_datasets(config, start_phase, end_phase):
    """
    Download datasets for the specified phases, verify integrity, and save them locally.
    """
    # Get the base path from the configuration, defaulting to '~/rachana_ddp' if not provided
    base_path = os.path.expanduser(config.get("base_path", "~/rachana_ddp"))
    data_dir = os.path.join(base_path, "data")
    os.makedirs(data_dir, exist_ok=True)

    logging.info("\nDownloading datasets...")
    for phase_num in range(start_phase, end_phase + 1):
        phase_key = f"phase{phase_num}"
        phase_info = config["phases"].get(phase_key)
        if phase_info:
            dataset_name = phase_info["dataset_name"]
            dataset_path = os.path.join(data_dir, f"phase{phase_num}")
            
            if os.path.isdir(dataset_path) and verify_dataset_integrity(dataset_path):
                logging.info(f"Dataset for Phase {phase_num} already exists and is verified.")
                continue
            
            logging.info(f"Downloading dataset '{dataset_name}' for Phase {phase_num}...")
            try:
                dataset = load_dataset(dataset_name, split="train")
                save_dataset_to_disk(dataset, dataset_path)
            except Exception as e:
                logging.error(f"Failed to download dataset {dataset_name} for Phase {phase_num}. Error: {e}")
        else:
            logging.error(f"Phase {phase_num} configuration not found in the config file.")


def load_model_config(config_path):
    """
    Load the model configuration from the config.json file.
    """
    with open(config_path, "r") as file:
        config = json.load(file)
    return config["model_config"]


def initialize_model(model_config):
    """
    Initialize the GPT-2 model based on the loaded configuration.
    """
    gpt2_config = GPT2Config(
        vocab_size=model_config['vocab_size'],
        n_positions=model_config['n_positions'],
        n_ctx=model_config['n_ctx'],
        n_embd=model_config['embedding_size'],
        n_layer=model_config['num_layers'],
        n_head=model_config['num_heads'],
        activation_function=model_config['activation_function'],
        resid_pdrop=model_config['dropout_rate'],
        embd_pdrop=model_config['dropout_rate'],
        attn_pdrop=model_config['dropout_rate'],
        initializer_range=model_config['initializer_range'],
        layer_norm_epsilon=model_config['layer_norm_epsilon'],
        pad_token_id=model_config['pad_token_id'],
        bos_token_id=model_config['bos_token_id'],
        eos_token_id=model_config['eos_token_id']
    )

    # Initialize the GPT-2 model with the configuration
    model = GPT2Model(gpt2_config)
    return model

def load_checkpoint(model, checkpoint_path, device):
    """
    Load the model checkpoint from the specified path if it exists.
    """
    if os.path.exists(checkpoint_path):
        logging.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint)
        logging.info(f"Checkpoint loaded successfully from {checkpoint_path}")
    else:
        logging.warning(f"No checkpoint found at {checkpoint_path}. Starting from scratch.")
    return model


def print_model_parameters(model):
    """
    Print the total, trainable, and embedding parameters of the model.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    embedding_params = sum(p.numel() for p in model.get_input_embeddings().parameters())

    logging.info(f"Total Parameters: {total_params/1e6:.2f}M")
    logging.info(f"Trainable Parameters: {trainable_params/1e6:.2f}M")
    logging.info(f"Embedding Parameters: {embedding_params/1e6:.2f}M")

# Custom collate function with improved error handling
def custom_collate_fn(batch, max_length=512):
    filtered_batch = []
    for x in batch:
        input_ids = x.get('input_ids', [])
        target_ids = x.get('target_ids') or [x.get('target_id')]

        if input_ids is None or target_ids is None or len(input_ids) == 0 or len(target_ids) == 0:
            continue
        
        if len(input_ids) <= max_length and len(target_ids) <= max_length:
            filtered_batch.append({'input_ids': input_ids, 'target_ids': target_ids})

    if not filtered_batch:
        return {}

    input_ids = [torch.tensor(x['input_ids'], dtype=torch.long) for x in filtered_batch]
    target_ids = [torch.tensor(x['target_ids'], dtype=torch.long) for x in filtered_batch]

    input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    target_ids_padded = torch.nn.utils.rnn.pad_sequence(target_ids, batch_first=True, padding_value=0)
    attention_masks = torch.nn.utils.rnn.pad_sequence([torch.ones(len(seq), dtype=torch.long) for seq in input_ids], batch_first=True, padding_value=0)

    return {
        'input_ids': input_ids_padded,
        'target_ids': target_ids_padded,
        'attention_mask': attention_masks
    }

# Function to load datasets and create DataLoader
def create_dataloader(dataset_path, batch_size=8):
    dataset = load_from_disk(dataset_path)
    sampler = distributed.DistributedSampler(dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        collate_fn=lambda x: custom_collate_fn(x, max_length=512)
    )

    return dataloader

def validate_model_with_data(model, dataloader, phase_name):
    """
    Validate the model with three random batches to ensure data and model shapes align.
    """
    model.eval()  # Set the model to evaluation mode
    for i, batch in enumerate(dataloader):
        if not batch or 'input_ids' not in batch or 'target_ids' not in batch:
            logging.info(f"Skipped an empty or invalid batch from {phase_name}.")
            continue

        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']

        # Run a forward pass through the model
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        logging.info(f"Batch {i+1} from {phase_name}: Model output shape {outputs.last_hidden_state.shape}")

        if i == 2:  # Test with three random batches
            break

def get_scheduler(optimizer, num_warmup_steps, num_training_steps):
    """
    Create a learning rate scheduler based on the optimizer and configuration parameters.
    """
    return get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

def train_with_mixed_precision_ddp(model, dataloader, device, optimizer, scheduler, scaler, writer, phase_name, num_epochs=1, gradient_accumulation_steps=1):
    model.train()
    total_loss = 0.0
    total_tokens = 0
    start_time = time.time()
    
    for epoch in range(num_epochs):
        logging.info(f"\nTraining {phase_name} - Epoch {epoch + 1}/{num_epochs}")
        running_loss = 0.0
        epoch_tokens = 0
        epoch_start_time = time.time()
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Phase {phase_name} - Epoch {epoch + 1}")):
            if not batch:
                continue

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            optimizer.zero_grad()

            with autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = outputs.last_hidden_state.mean()  # Dummy loss calculation, replace with actual loss
                loss = loss / gradient_accumulation_steps

            scaler.scale(loss).backward()

            if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

            running_loss += loss.item()
            epoch_tokens += input_ids.numel()

            if (batch_idx + 1) % 10 == 0 and dist.get_rank() == 0:  # Log every 10 batches on main process
                avg_loss = running_loss / 10
                writer.add_scalar(f'{phase_name}/loss', avg_loss, epoch * len(dataloader) + batch_idx)
                running_loss = 0.0

        epoch_duration = time.time() - epoch_start_time
        total_loss += running_loss
        total_tokens += epoch_tokens

        if dist.get_rank() == 0:  # Log epoch completion on main process
            logging.info(f"Completed Epoch {epoch + 1} of {phase_name}")
            logging.info(f"Epoch Duration: {timedelta(seconds=int(epoch_duration))}, Tokens Processed: {epoch_tokens}")
    
    total_time = time.time() - start_time
    logging.info(f"Training Completed for {phase_name}. Total Duration: {timedelta(seconds=int(total_time))}, Total Tokens: {total_tokens}")
    log_gpu_memory(device.index)

# Main function to execute the training pipeline
def main():
    args = parse_arguments()
    start_phase = args.start_phase
    end_phase = args.end_phase

    rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    setup(rank, world_size)
    
    # Construct paths dynamically based on the home directory and base path from config
    home_dir = os.path.expanduser("~")

    # Load configuration
    config_path = os.path.join(home_dir, "rachana_ddp/config/config.json")
    if not os.path.exists(config_path):
        logging.error(f"Configuration file not found at: {config_path}")
        return
    
    config = load_config(config_path)
    base_path = os.path.expanduser(config.get("base_path", "~/rachana_ddp"))  # Expand base path

    # Set paths
    data_dir = os.path.join(base_path, "data")
    log_dir = os.path.join(base_path, "logs/tensorboard")
    
    # Ensure directories exist
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Ensure logging only happens on the master node
    writer = SummaryWriter(log_dir) if dist.get_rank() == 0 else None

    logging.info(f"Configuration file loaded from: {config_path}")
    
    if start_phase < config['training_control']['start_phase'] or end_phase > config['training_control']['end_phase']:
        logging.error(f"Invalid phase range. Please enter a range between {config['training_control']['start_phase']} and {config['training_control']['end_phase']}.")
        return
    
    print_phase_info(config, start_phase, end_phase)
    download_datasets(config, start_phase, end_phase)

    # Load model configuration and initialize model
    model_config = load_model_config(config_path)
    model = initialize_model(model_config)
    
    # Print model parameters
    print_model_parameters(model)

    device = torch.device("cuda:0")  # Use device 0 explicitly
    model = model.to(device)
    model = DDP(model, device_ids=[0])  # Wrap with DistributedDataParallel

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["phases"]["phase1"]["learning_rate"])
    num_training_steps = sum([config["phases"][f"phase{p}"]["epochs"] * len(create_dataloader(os.path.join(data_dir, f"phase{p}"), config["phases"][f"phase{p}"]["batch_size"])) for p in range(start_phase, end_phase + 1)])
    scheduler = get_scheduler(optimizer, num_warmup_steps=config["phases"]["phase1"]["warmup_steps"], num_training_steps=num_training_steps)
    scaler = GradScaler()

    phase_paths = {f"Phase {i}": os.path.join(data_dir, f"phase{i}") for i in range(start_phase, end_phase + 1)}

    # Validate model alignment with data before training
    all_aligned = True
    for phase_name, phase_path in phase_paths.items():
        dataloader = create_dataloader(phase_path, config["phases"][phase_name.lower().replace(" ", "")]["batch_size"])
        try:
            validate_model_with_data(model, dataloader, phase_name)
        except Exception as e:
            logging.error(f"Validation failed for {phase_name}: {e}")
            all_aligned = False

    if not all_aligned:
        logging.error("\nSome datasets are not aligned with the model configuration. Please check the logs for details.")
        return

    # Train for each phase in the specified range
    for phase_num in range(start_phase, end_phase + 1):
        phase_name = f"Phase {phase_num}"
        phase_path = phase_paths[phase_name]
        phase_info = config["phases"][phase_name.lower().replace(" ", "")]
        gradient_accumulation_steps = phase_info["gradient_accumulation_steps"]
        checkpoint_dir = os.path.join(base_path, phase_info["checkpoint_dir"])  # Dynamic checkpoint path

        logging.info(f"\nStarting training for {phase_name}")
        dataloader = create_dataloader(phase_path, phase_info["batch_size"])

        # Load checkpoint if available
        if start_phase > 1:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_phase_{phase_num - 1}.pt")
            model = load_checkpoint(model, checkpoint_path, device)

        train_with_mixed_precision_ddp(
            model, dataloader, device, optimizer, scheduler, scaler, writer, phase_name,
            num_epochs=phase_info["epochs"], gradient_accumulation_steps=gradient_accumulation_steps
        )

        # Save checkpoint at the end of each phase
        if dist.get_rank() == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_phase_{phase_num}.pt")
            torch.save(model.state_dict(), checkpoint_path)
            logging.info(f"Checkpoint saved for {phase_name} at {checkpoint_path}")

    if dist.get_rank() == 0:
        logging.info("\nAll phases completed successfully. Model is ready for next steps!")

    # Cleanup DDP
    dist.barrier()
    dist.destroy_process_group()


