#!/bin/bash
cd /mnt/e/developer/LeWorldModel
source ~/.venv_procgen/bin/activate

echo "================================================="
echo "1. Training PPO Agent on CoinRun (1M timesteps)..."
echo "================================================="
python3 src/train_agent.py --env procgen:procgen-coinrun-v0 --timesteps 1000000

echo "================================================="
echo "2. Collecting Expert Trajectories (100 episodes)..."
echo "================================================="
python3 src/data/collect_data.py --env procgen:procgen-coinrun-v0 --episodes 100 --model_path "models/ppo_procgen:procgen_coinrun_v0.zip"

echo "================================================="
echo "3. Training LeWM Model (100 epochs)..."
echo "================================================="
python3 src/train_lewm.py --env procgen:procgen-coinrun-v0 --epochs 100

echo "================================================="
echo "4. Evaluating OOD Detection..."
echo "================================================="
python3 src/evaluate_ood.py --env procgen:procgen-coinrun-v0 --steps 1000 --ood_step 500 --seeds 1

echo "================================================="
echo "PIPELINE COMPLETED SUCCESSFULLY!"
echo "================================================="
