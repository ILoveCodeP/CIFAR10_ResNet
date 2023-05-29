#!/bin/bash
#SBATCH -J ResCifarSingle       # job name, optional
#SBATCH -N 1          # number of computing node
#SBATCH -c 1          # number of cpus, for multi-thread programs
#SBATCH --gres=gpu:1  # number of gpus allocated on each node
#SBATCH -w node04  # apply for node05
#SBATCH -o ResCifarStructure.out

# 定义要运行的脚本参数
params=(
  "--num_filters=8"
  "--num_filters=16"
  "--num_filters=32"
  "--num_filters=64"
)

# 循环运行脚本
for param in "${params[@]}"; do
  python -u train.py $params 2>&1
  echo "Script completed: $param"
done
