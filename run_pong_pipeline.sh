#!/bin/bash
cd /mnt/e/developer/LeWorldModel
source ~/.venv_procgen/bin/activate

echo "================================================="
echo "1. Training PPO Agent on Pong..."
echo "(SKIPPED - Sử dụng mô hình đã có sẵn)"
echo "================================================="
# python3 src/train_agent.py --env ALE/Pong-v5 --timesteps 1000000

echo "================================================="
echo "2. Collecting Expert Trajectories (50 episodes)..."
echo "================================================="
python3 src/data/collect_data.py --env ALE/Pong-v5 --episodes 50 --model_path "models/ppo_ALE_Pong_v5.zip"

echo "================================================="
echo "3. Training LeWM Model (50 epochs)..."
echo "================================================="
python3 src/train_lewm.py --env ALE/Pong-v5 --epochs 50

echo "================================================="
echo "4. Evaluating OOD Detection..."
echo "================================================="
python3 src/evaluate_ood.py --env ALE/Pong-v5 --steps 1000 --ood_step 500 --seeds 5

echo "================================================="
echo "PIPELINE PONG COMPLETED SUCCESSFULLY!"
echo "================================================="
