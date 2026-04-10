#!/bin/bash

python eval.py \
  --model tensorboard/PPO_20250907_024039_idtirl_ppo/model_100000_steps.zip \
  --config tirl_ppo \
  --port 2020 \
  --device cuda:0 \
  --no_record_video \
  --town Town02 \
  --density regular