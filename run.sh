#!/bin/bash
#SBATCH --job-name=icg
#SBATCH --nodes=1
#SBATCH --partition=batch       # Specify the partition
#SBATCH --constraint=type-gpu   # Or type-gpu for GPU nodes
#SBATCH --mem=30G              # Memory limit
#SBATCH --mail-type=ALL         # Send updates via Slack
#SBATCH --time=3-0		          # Time limit

#SBATCH --output=/scratch/%u/logs/icg.log
#SBATCH --error=/scratch/%u/logs/icg.error
#SBATCH --gpus h200:1

user="$USER"

export MAMBA_ROOT_PREFIX="/scratch/$USER/micromamba"
export PATH="$MAMBA_ROOT_PREFIX/bin:$PATH"

mkdir -p $SLURM_SUBMIT_DIR
rsync -aHzv --exclude='.*' ${SLURM_SUBMIT_HOST}:"${SLURM_SUBMIT_DIR}"/ "$(rsync-path $SLURM_SUBMIT_DIR)"
cd $SLURM_SUBMIT_DIR

num_gpus=$(echo "$SLURM_GPUS_ON_NODE" | grep -o '[0-9]*' | head -1)
num_gpus="${num_gpus:-1}"
cuda_devices=$(seq -s, 0 $((num_gpus - 1)))

# Set the CUDA_VISIBLE_DEVICES in vLLM-friendly format
export CUDA_VISIBLE_DEVICES="$cuda_devices"

echo $PATH

eval "$(micromamba shell hook --shell=bash)"

micromamba create -n icg -f env.yml -y
micromamba activate icg

srun python -u train.py