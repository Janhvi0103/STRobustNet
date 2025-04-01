#!/usr/bin/env bash
#SBATCH -J str
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -o multistage-noise-bs32-epo50.%j
#SBATCH -e multistage-noise-bs32-epo50.%j
#SBATCH --nodelist=gpu4

#!/usr/bin/env bash

gpus=0

data_name=SYSU
net_G=STR
split=test
project_name=STR-sysu
hidden_dim=128
dec_layers=3
drop_out=0
num_queries=20

python eval_cd.py --num_queries ${num_queries} --drop_out ${drop_out} --dec_layers ${dec_layers} --hidden_dim ${hidden_dim} --split ${split} --net_G ${net_G} --gpu_ids ${gpus} --project_name ${project_name} --data_name ${data_name}


