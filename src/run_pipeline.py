import subprocess
import sys
import os

def run_cmd(cmd):
    print(f"\\n[Pipeline] Running command: {cmd}")
    # Force UTF-8 environment for subprocess output logging
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    
    res = subprocess.run(cmd, shell=True, env=env)
    if res.returncode != 0:
        print(f"[Pipeline] Command failed with return code {res.returncode}: {cmd}")
        sys.exit(1)

if __name__ == "__main__":
    # Đặt thư mục làm việc về gốc dự án
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(current_dir)
    os.chdir(project_dir)
    
    print("================ BẮT ĐẦU PIPELINE TỰ ĐỘNG ==================")
    print("Lưu ý: Có thể sửa các lệnh dưới đây thành --env procgen:procgen-coinrun-v0 để chạy với Procgen.")
    print("1. Huấn luyện PPO Agent (300.000 timesteps)...")
    run_cmd(r".\.venv\Scripts\python.exe src/train_agent.py --env ALE/Pong-v5 --timesteps 300000")
    
    print("\n2. Thu thập dữ liệu hành trình (10 episodes)...")
    run_cmd(r".\.venv\Scripts\python.exe src/data/collect_data.py --env ALE/Pong-v5 --episodes 10 --model_path models/ppo_ALE_Pong_v5.zip")
    
    print("\n3. Huấn luyện LeWorldModel & Pixel Baseline (15 epochs)...")
    run_cmd(r".\.venv\Scripts\python.exe src/train_lewm.py --env ALE/Pong-v5 --epochs 15")
    
    print("\n4. Chạy đánh giá OOD Dynamics (Frameskip 4 -> 2 tại step 200)...")
    run_cmd(r".\.venv\Scripts\python.exe src/detect_ood.py --env ALE/Pong-v5 --steps 400 --ood_step 200 --new_frameskip 2")
    
    print("\n================ PIPELINE HOÀN TẤT THÀNH CÔNG ==============")
