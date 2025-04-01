#!/usr/bin/env bash

gpus=0
checkpoint_root=checkpoints
data_name=WHU

img_size=256
batch_size=16
lr=0.01
max_epochs=200
net_G=STR
lr_policy=linear
num_queries=4
input_channels=3
warm_up=0
num_workers=16
hidden_dim=128
dec_layers=3

split=train
split_val=val
project_name=STR-whu

CUDA_VISIBLE_DEVICES=0 python -u main_cd.py --dec_layers ${dec_layers} --hidden_dim ${hidden_dim} --num_workers ${num_workers} --input_channels ${input_channels} --num_queries ${num_queries} --img_size ${img_size} --checkpoint_root ${checkpoint_root} --lr_policy ${lr_policy} --split ${split} --split_val ${split_val} --net_G ${net_G} --gpu_ids ${gpus} --max_epochs ${max_epochs} --project_name ${project_name} --batch_size ${batch_size} --data_name ${data_name}  --lr ${lr}