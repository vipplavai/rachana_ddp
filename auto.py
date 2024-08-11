import paramiko
import subprocess
import time
import os
import psutil
import threading
import datetime
import csv

logs = {
    "master": ""
}

def kill_process_on_port(port):
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            for conns in proc.connections(kind='inet'):
                if conns.laddr.port == port:
                    print(f"Killing process {proc.info['name']} with PID {proc.info['pid']} using port {port}")
                    proc.terminate()
                    proc.wait()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

def add_timestamped_log(log_key, message):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if log_key not in logs:
        logs[log_key] = ""
    if isinstance(logs[log_key], dict) and 'log' in logs[log_key]:
        logs[log_key]['log'] += f"[{timestamp}] {message}\n"
    else:
        logs[log_key] += f"[{timestamp}] {message}\n"

def stream_output(process, log_key):
    for output in iter(process.stdout.readline, ''):
        if output:
            add_timestamped_log(log_key, output.strip())
            print(output, end='')

def run_torchrun_locally(script_path, rank, world_size, master_addr, master_port, log_key):
    env = os.environ.copy()
    env['RANK'] = str(rank)
    env['WORLD_SIZE'] = str(world_size)
    env['MASTER_ADDR'] = master_addr
    env['MASTER_PORT'] = str(master_port)
    env['CUDA_VISIBLE_DEVICES'] = '0'
    env['CUDA_HOME'] = '/usr/local/cuda'
    env['PATH'] = '/usr/local/cuda/bin:' + env['PATH']
    env['LD_LIBRARY_PATH'] = '/usr/local/cuda/lib64:' + env.get('LD_LIBRARY_PATH', '')
    command = [
        'torchrun',
        '--nproc_per_node=1',
        '--nnodes={}'.format(world_size),
        '--node_rank={}'.format(rank),
        '--master_addr={}'.format(master_addr),
        '--master_port={}'.format(master_port),
        script_path
    ]
    process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, universal_newlines=True)
    threading.Thread(target=stream_output, args=(process, log_key)).start()
    return process

def run_torchrun_remotely(hostname, username, script_path, rank, world_size, master_addr, master_port, conda_env, log_key):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        ssh.connect(hostname, username=username)
    except Exception as e:
        error_message = f"SSH connection to {hostname} failed: {e}"
        add_timestamped_log(log_key, error_message)
        print(error_message)
        return None

    command = f"""
    source ~/miniconda3/bin/activate {conda_env} && \
    export RANK={rank} && \
    export WORLD_SIZE={world_size} && \
    export MASTER_ADDR={master_addr} && \
    export MASTER_PORT={master_port} && \
    export CUDA_VISIBLE_DEVICES=0 && \
    export CUDA_HOME=/usr/local/cuda && \
    export PATH=/usr/local/cuda/bin:$PATH && \
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH && \
    torchrun --nproc_per_node=1 --nnodes={world_size} --node_rank={rank} --master_addr={master_addr} --master_port={master_port} {script_path}
    """
    stdin, stdout, stderr = ssh.exec_command(command)

    def stream_ssh_output(stream, log_key):
        for line in iter(stream.readline, ''):
            if line:
                add_timestamped_log(log_key, line.strip())
                print(line, end='')

    threading.Thread(target=stream_ssh_output, args=(stdout, log_key)).start()
    threading.Thread(target=stream_ssh_output, args=(stderr, log_key)).start()

    return ssh

def read_nodes_from_csv(file_path):
    master = None
    workers = []
    with open(file_path, mode='r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            if row["role"] == "master":
                master = {"hostname": row["hostname"], "username": row["username"]}
            elif row["role"] == "worker":
                workers.append({"hostname": row["hostname"], "username": row["username"]})
    return master, workers

if __name__ == "__main__":
    nodes_csv = "nodes.csv"
    master, workers = read_nodes_from_csv(nodes_csv)

    if not master:
        print("No master node found in the CSV file.")
        exit(1)

    conda_env = "ddp_setup"
    script_path = "/home/{username}/rachana_ddp/train.py"
    master_addr = master["hostname"]
    master_port = 29500
    world_size = 1 + len(workers)  # 1 master + number of workers

    # Ensure logs directory exists
    os.makedirs('logs', exist_ok=True)

    # Initialize logs for workers
    logs['workers'] = []
    for i, worker in enumerate(workers, start=1):
        worker_key = f"worker_{i}"
        logs[worker_key] = {"log": ""}  # Properly initialize log entry for each worker
        logs['workers'].append({
            "key": worker_key,
            "name": f"Worker {i}",
            "username": worker["username"],
            "log": ""
        })

    # Kill any process using the port on the master node
    kill_process_on_port(master_port)

    # Run script on master node (rank 0) in a separate thread
    logs['master'] = {
        "username": master["username"],
        "log": ""
    }
    master_script_path = script_path.format(username=master["username"])
    master_process = run_torchrun_locally(master_script_path, rank=0, world_size=world_size, master_addr=master_addr, master_port=master_port, log_key="master")

    # Allow some time for the master node to set up
    time.sleep(5)

    # Run script on each worker node in separate threads
    worker_threads = []
    for i, worker in enumerate(workers, start=1):
        worker_script_path = script_path.format(username=worker["username"])
        thread = threading.Thread(
            target=run_torchrun_remotely,
            args=(worker["hostname"], worker["username"], worker_script_path, i, world_size, master_addr, master_port, conda_env, f"worker_{i}")
        )
        worker_threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in worker_threads:
        thread.join()

    # Wait for master process to complete
    master_process.wait()
