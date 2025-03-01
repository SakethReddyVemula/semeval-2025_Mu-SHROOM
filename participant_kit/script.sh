#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=2048
#SBATCH --time=24:00:00
#SBATCH --output=baseline_model.txt
#SBATCH --mail-user=saketh.vemula@research.iiit.ac.in
#SBATCH --mail-type=ALL
#SBATCH --nodelist=gnode082  # Request specific node

# Activate your virtual environment
source /home2/saketh.vemula/ltg_venv/bin/activate

get_free_port() {
    python -c "import socket; s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.bind(('', 0)); port = s.getsockname()[1]; s.close(); print(port)"
}

export MASTER_PORT=$(get_free_port)
echo "MASTER_PORT="$MASTER_PORT

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR 

# Optional: Set up any necessary environment variables
export WANDB_API_KEY="c8a7a539cb5fed3df89b21d71956ca6b4befd2a5" # Set api key of wandb in script
export WANDB_PROJECT="semeval-2025"

# Run the training script
torchrun  \
    --master_port $MASTER_PORT \
    --nproc_per_node 4 \
    --nnodes 1 \
baseline_model.py \
    --mode test \
    --model_checkpoint ./results/checkpoint-145 \
    --data_path ../val \
    --test_lang en
