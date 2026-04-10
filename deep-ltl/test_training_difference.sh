#!/bin/bash

# 快速测试：运行 PPO 和 PPO-LTL 各 2000 步，看看结果是否不同

set -e

PROJECT_ROOT=$(cd "$(dirname "$0")" && pwd)
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT/CRL:${PYTHONPATH:-}"

SEED=999
STEPS=5000

echo "========== 测试 1: PPO (no constraint) =========="
python CRL/train_ppo.py \
  --env ZoneEnv-v0 \
  --exp "test_ppo_seed${SEED}" \
  --save-dir "$PROJECT_ROOT/test_runs" \
  --algorithm PPO \
  --seed "$SEED" \
  --n-envs 1 \
  --total-steps "$STEPS" \
  --device cpu \
  --sync-env

echo ""
echo "========== 测试 2: PPO-LTL (with constraint) =========="
python CRL/train_ppo.py \
  --env ZoneEnv-v0 \
  --exp "test_ppoltl_seed${SEED}" \
  --save-dir "$PROJECT_ROOT/test_runs" \
  --algorithm PPOLag \
  --seed "$SEED" \
  --n-envs 1 \
  --total-steps "$STEPS" \
  --formula "(!blue U green) & F yellow" \
  --finite \
  --cost-limit 0.05 \
  --lagrangian-lr 0.005 \
  --device cpu \
  --sync-env

echo ""
echo "========== 对比结果 =========="
echo "PPO episodes.csv:"
tail -5 "$PROJECT_ROOT/test_runs/test_ppo_seed${SEED}/episodes.csv"
echo ""
echo "PPO-LTL episodes.csv:"
tail -5 "$PROJECT_ROOT/test_runs/test_ppoltl_seed${SEED}/episodes.csv"
echo ""
echo "检查是否相同:"
if diff "$PROJECT_ROOT/test_runs/test_ppo_seed${SEED}/episodes.csv" "$PROJECT_ROOT/test_runs/test_ppoltl_seed${SEED}/episodes.csv" > /dev/null 2>&1; then
    echo "❌ 完全相同！有bug！"
else
    echo "✅ 不相同，正常"
fi
