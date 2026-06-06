import torch
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import argparse
import sys
from stable_baselines3 import PPO
import ale_py
import gymnasium as gym

# Thêm đường dẫn project vào sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
if project_dir not in sys.path:
    sys.path.append(project_dir)

from src.env.atari_ood_wrapper import AtariDynamicFrameskipWrapper
from src.models.lewm import LeWorldModel
from src.models.baseline import PixelPredictor

def compute_auroc(y_true, y_scores):
    """
    Tính toán AUROC sử dụng công thức Wilcoxon-Mann-Whitney (không dùng sklearn).
    """
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    pos = y_scores[y_true == 1]
    neg = y_scores[y_true == 0]
    
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
        
    pos = np.sort(pos)
    neg = np.sort(neg)
    
    # Tìm vị trí chèn của neg trong pos
    idx = np.searchsorted(pos, neg)
    return np.sum(len(pos) - idx) / (len(pos) * len(neg))

def compute_max_f1(y_true, y_scores):
    """
    Tính toán F1-Score tốt nhất bằng cách duyệt qua các ngưỡng phân ngưỡng.
    """
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    best_f1 = 0.0
    thresholds = np.linspace(y_scores.min(), y_scores.max(), 100)
    for t in thresholds:
        preds = (y_scores >= t).astype(int)
        tp = np.sum((preds == 1) & (y_true == 1))
        fp = np.sum((preds == 1) & (y_true == 0))
        fn = np.sum((preds == 0) & (y_true == 1))
        
        if tp + fp == 0 or tp + fn == 0:
            continue
            
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        
        if precision + recall == 0:
            continue
            
        f1 = 2 * (precision * recall) / (precision + recall)
        if f1 > best_f1:
            best_f1 = f1
    return best_f1

def preprocess_frame(obs):
    """
    Chuyển đổi sang ảnh xám và resize về 84x84.
    """
    if len(obs.shape) == 3 and obs.shape[-1] == 3:
        obs_gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    else:
        obs_gray = obs
    obs_resized = cv2.resize(obs_gray, (84, 84), interpolation=cv2.INTER_AREA)
    return obs_resized

def run_ood_evaluation(env_id="ALE/Pong-v5", steps=400, ood_step=200, new_frameskip=2, latent_dim=64):
    clean_env_id = env_id.replace("/", "_").replace("-", "_")
    
    # Đường dẫn tải models
    models_dir = os.path.join(project_dir, "models")
    ppo_path = os.path.join(models_dir, f"ppo_{clean_env_id}.zip")
    lewm_path = os.path.join(models_dir, f"lewm_{clean_env_id}.pth")
    baseline_path = os.path.join(models_dir, f"baseline_{clean_env_id}.pth")
    
    # 1. Khởi tạo môi trường test
    print(f"Khởi tạo môi trường thử nghiệm {env_id}...")
    env = gym.make(env_id, frameskip=1, render_mode="rgb_array")
    env = AtariDynamicFrameskipWrapper(env, initial_frameskip=4)
    
    # Xác định số lượng hành động
    action_dim = 6
    if "Breakout" in env_id:
        action_dim = 4
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. Tải PPO Agent để tạo hành động thông minh
    ppo_model = None
    if os.path.exists(ppo_path):
        print(f"Đang tải PPO Agent từ {ppo_path}...")
        ppo_model = PPO.load(ppo_path)
    else:
        print("Cảnh báo: Không tìm thấy PPO Agent. Sẽ dùng Random Agent để điều khiển!")
        
    # 3. Tải LeWorldModel và Baseline
    if not os.path.exists(lewm_path) or not os.path.exists(baseline_path):
        print("Lỗi: Không tìm thấy trọng số đã huấn luyện của LeWM hoặc Baseline!")
        print("Vui lòng chạy train_lewm.py trước.")
        return
        
    print("Đang tải LeWorldModel...")
    lewm = LeWorldModel(latent_dim=latent_dim, action_dim=action_dim).to(device)
    lewm.load_state_dict(torch.load(lewm_path, map_location=device))
    lewm.eval()
    
    print("Đang tải Pixel Predictor Baseline...")
    baseline = PixelPredictor(action_dim=action_dim).to(device)
    baseline.load_state_dict(torch.load(baseline_path, map_location=device))
    baseline.eval()
    
    # 4. Tạo bộ đệm lưu trữ
    latent_surprises = []
    pixel_surprises = []
    labels = [] # 0 cho ID, 1 cho OOD
    
    # Bộ đệm Frame Stack (4 frames) cho PPO
    frame_stack = np.zeros((1, 4, 84, 84), dtype=np.uint8)
    
    def update_frame_stack(frame):
        frame_stack[0, :-1] = frame_stack[0, 1:]
        frame_stack[0, -1] = frame

    obs, info = env.reset()
    obs_processed = preprocess_frame(obs)
    update_frame_stack(obs_processed)
    
    print(f"\nBắt đầu chạy thử nghiệm với {steps} bước...")
    
    with torch.no_grad():
        for step in range(steps):
            # Chọn hành động
            if ppo_model is not None:
                action, _ = ppo_model.predict(frame_stack, deterministic=False)
                action_val = int(action[0])
            else:
                action_val = env.action_space.sample()
                
            # Ghi nhận trạng thái hiện tại (s_t, a_t)
            obs_t_tensor = torch.tensor(obs_processed, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device) / 255.0
            act_t_tensor = torch.tensor([action_val], dtype=torch.long).to(device)
            
            # Kích hoạt OOD Dynamics ở step chỉ định
            if step == ood_step:
                env.trigger_ood(new_frameskip=new_frameskip)
                
            # Thực hiện hành động
            next_obs, reward, terminated, truncated, info = env.step(action_val)
            done = terminated or truncated
            
            # Ghi nhận nhãn OOD
            is_ood = 1 if step >= ood_step else 0
            labels.append(is_ood)
            
            # Tiền xử lý next_obs
            next_obs_processed = preprocess_frame(next_obs)
            next_obs_tensor = torch.tensor(next_obs_processed, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device) / 255.0
            
            # Cập nhật Frame Stack cho PPO
            update_frame_stack(next_obs_processed)
            
            # --- Tính Latent Surprise (LeWM) ---
            z_t = lewm.get_latent(obs_t_tensor)
            z_t1 = lewm.get_latent(next_obs_tensor)
            pred_z_t1 = lewm.predict_next(z_t, act_t_tensor)
            l_surprise = torch.mean((pred_z_t1 - z_t1) ** 2).item()
            latent_surprises.append(l_surprise)
            
            # --- Tính Pixel Surprise (Baseline) ---
            pred_obs_t1 = baseline(obs_t_tensor, act_t_tensor)
            p_surprise = torch.mean((pred_obs_t1 - next_obs_tensor) ** 2).item()
            pixel_surprises.append(p_surprise)
            
            obs_processed = next_obs_processed
            
            if done:
                # Nếu game kết thúc trước hạn, reset lại môi trường nhưng giữ nguyên trạng thái OOD
                current_fs = env.frameskip
                obs, info = env.reset()
                env.frameskip = current_fs
                obs_processed = preprocess_frame(obs)
                update_frame_stack(obs_processed)
                
    env.close()
    
    # Convert sang numpy array
    latent_surprises = np.array(latent_surprises)
    pixel_surprises = np.array(pixel_surprises)
    labels = np.array(labels)
    
    # 5. Chuẩn hóa Surprise Scores sử dụng thống kê từ giai đoạn ID (steps < 150)
    id_cutoff = min(150, ood_step)
    
    lewm_mean = np.mean(latent_surprises[:id_cutoff])
    lewm_std = np.std(latent_surprises[:id_cutoff]) + 1e-6
    norm_latent_surprises = (latent_surprises - lewm_mean) / lewm_std
    
    base_mean = np.mean(pixel_surprises[:id_cutoff])
    base_std = np.std(pixel_surprises[:id_cutoff]) + 1e-6
    norm_pixel_surprises = (pixel_surprises - base_mean) / base_std
    
    # 6. Đánh giá Metrics
    lewm_auroc = compute_auroc(labels, norm_latent_surprises)
    lewm_f1 = compute_max_f1(labels, norm_latent_surprises)
    
    base_auroc = compute_auroc(labels, norm_pixel_surprises)
    base_f1 = compute_max_f1(labels, norm_pixel_surprises)
    
    print("\n================ KẾT QUẢ ĐÁNH GIÁ OOD DYNAMICS ================")
    print(f"Môi trường: {env_id} | Thay đổi frameskip sang: {new_frameskip} ở step {ood_step}")
    print(f"1. LeWorldModel (JEPA):")
    print(f"   - AUROC:  {lewm_auroc:.4f}")
    print(f"   - Max F1: {lewm_f1:.4f}")
    print(f"2. Pixel Predictor (Baseline):")
    print(f"   - AUROC:  {base_auroc:.4f}")
    print(f"   - Max F1: {base_f1:.4f}")
    print("==============================================================")
    
    # 7. Trực quan hóa và vẽ biểu đồ
    os.makedirs(os.path.join(project_dir, "results"), exist_ok=True)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Subplot 1: LeWorldModel Latent Surprise
    axes[0].plot(norm_latent_surprises, label="Normalized Latent Surprise (LeWM)", color="#1f77b4", linewidth=2)
    axes[0].axvline(x=ood_step, color="red", linestyle="--", label="OOD Dynamics Trigger", linewidth=2)
    axes[0].set_title("LeWorldModel (JEPA) OOD Detection", fontsize=14, fontweight='bold')
    axes[0].set_ylabel("Normalized Surprise Score", fontsize=12)
    axes[0].legend(loc="upper left")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    
    # Subplot 2: Pixel Predictor Surprise
    axes[1].plot(norm_pixel_surprises, label="Normalized Pixel Surprise (Baseline)", color="#ff7f0e", linewidth=2)
    axes[1].axvline(x=ood_step, color="red", linestyle="--", label="OOD Dynamics Trigger", linewidth=2)
    axes[1].set_title("Pixel Reconstruction (Baseline) OOD Detection", fontsize=14, fontweight='bold')
    axes[1].set_xlabel("Time Step", fontsize=12)
    axes[1].set_ylabel("Normalized Surprise Score", fontsize=12)
    axes[1].legend(loc="upper left")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    
    plt.tight_layout()
    plot_save_path = os.path.join(project_dir, "results", f"ood_detection_{clean_env_id}.png")
    plt.savefig(plot_save_path, dpi=150)
    print(f"\n[Thành công] Đã lưu biểu đồ so sánh tại:\n {plot_save_path}")
    
    # Lưu file log text kết quả
    log_save_path = os.path.join(project_dir, "results", f"metrics_{clean_env_id}.txt")
    with open(log_save_path, "w", encoding="utf-8") as f:
        f.write(f"Environment: {env_id}\n")
        f.write(f"OOD Step: {ood_step}\n")
        f.write(f"New Frameskip: {new_frameskip}\n")
        f.write(f"LeWM AUROC: {lewm_auroc:.6f}\n")
        f.write(f"LeWM F1: {lewm_f1:.6f}\n")
        f.write(f"Baseline AUROC: {base_auroc:.6f}\n")
        f.write(f"Baseline F1: {base_f1:.6f}\n")
    print(f"Đã lưu kết quả text tại: {log_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate OOD Dynamics Detection using LeWM vs Baseline")
    parser.add_argument("--env", type=str, default="ALE/Pong-v5", help="Atari Environment ID")
    parser.add_argument("--steps", type=int, default=400, help="Total evaluation steps")
    parser.add_argument("--ood_step", type=int, default=200, help="Step at which OOD is triggered")
    parser.add_argument("--new_frameskip", type=int, default=2, help="New frameskip value for OOD")
    parser.add_argument("--latent_dim", type=int, default=64, help="Latent space dimension for LeWM")
    args = parser.parse_args()
    
    run_ood_evaluation(
        env_id=args.env,
        steps=args.steps,
        ood_step=args.ood_step,
        new_frameskip=args.new_frameskip,
        latent_dim=args.latent_dim
    )
