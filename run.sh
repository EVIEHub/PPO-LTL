#!/bin/bash

python train.py \
    --config=tirl_ppo \
    --start_carla \
    --no_render \
    --total_timesteps=1_000_000 \
    --port=2000 \
    --device=cuda:0