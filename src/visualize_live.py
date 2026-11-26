import argparse
import json
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import torch
import torch.nn as nn

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
if project_dir not in sys.path:
    sys.path.append(project_dir)

from src.env.atari_ood_wrapper import AtariDynamicFrameskipWrapper
from src.env.procgen_ood_wrapper import ProcgenOODWrapper
from src.models.lewm import LeWorldModel
from src.models.baseline import PixelPredictor

class OnlineEMANormalizer:
    def __init__(self, alpha: float = 0.99, eps: float = 1e-6):
        self.alpha = alpha
        self.eps = eps
        self.mu: float = 0.0
        self.var: float = 1.0
        self._initialized: bool = False
        
    def update_and_normalize(self, value: float, is_id: bool) -> float:
        if is_id:
            if not self._initialized:
                self.mu = value
                self.var = 1.0
                self._initialized = True
            else:
                self.mu = self.alpha * self.mu + (1 - self.alpha) * value
                self.var = self.alpha * self.var + (1 - self.alpha) * (value - self.mu) ** 2
        sigma = np.sqrt(self.var) + self.eps
        return (value - self.mu) / sigma


try:
    import gymnasium as gym
    import ale_py
except ImportError as e:
    raise ImportError(f"Missing dependency: {e}")

try:
    from stable_baselines3 import PPO
    HAS_PPO = True
except ImportError:
    HAS_PPO = False


def preprocess_frame(obs: np.ndarray) -> np.ndarray:
    if obs.ndim == 3 and obs.shape[-1] == 3:
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    return cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)

def plot_to_img(fig, ax, l_hist, p_hist, ood_step, current_step):
    ax.clear()
    ax.plot(l_hist, color="#1f77b4", label="LeWM (Latent)")
    ax.plot(p_hist, color="#ff7f0e", label="Baseline (Pixel)")
    
    if current_step >= ood_step:
        ax.axvspan(ood_step, current_step, color='red', alpha=0.2, label='OOD Region')
        
    ax.set_ylim(0, max(max(l_hist + p_hist + [1.0]), 2.5))
    ax.set_xlim(0, max(300, current_step))
    ax.set_title("Normalized Surprise Score")
    ax.legend(loc="upper left")
    
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())
    return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

def run_live_visualization(env_name="pong"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Settings based on env
    if env_name == "pong":
        env_id = "ALE/Pong-v5"
        base_frameskip = 4
        new_frameskip = 8
        ood_step = 100
        max_steps = 300
        is_procgen = False
        model_env_id = "ALE_Pong_v5"
    else:
        env_id = "procgen:procgen-coinrun-v0"
        base_frameskip = 1
        new_frameskip = 1
        ood_step = 100
        max_steps = 300
        is_procgen = True
        model_env_id = "procgen:procgen_coinrun_v0" # Sửa lại tên file procgen theo đúng chuẩn trong models/

    # 1. Setup Environment
    if is_procgen:
        env = gym.make(env_id, render_mode="rgb_array")
        env = ProcgenOODWrapper(env, base_momentum_prob=0.0)
    else:
        env = gym.make(env_id, render_mode="rgb_array", frameskip=1, repeat_action_probability=0.0)
        env = AtariDynamicFrameskipWrapper(env, initial_frameskip=base_frameskip)

    # 2. Load Models
    if is_procgen:
        lewm = LeWorldModel(action_dim=env.action_space.n, embed_dim=192).to(device)
    else:
        from src.models.lewm_cnn import CNNLeWorldModel
        lewm = CNNLeWorldModel(action_dim=env.action_space.n, embed_dim=64).to(device)
    
    # Tìm kiếm tên file (xử lý lỗi dấu hai chấm trong file name của Windows)
    lewm_filename = f"lewm_vit_procgen?procgen_coinrun_v0.pth" if is_procgen else f"lewm_{model_env_id}.pth"
    lewm_path = os.path.join(project_dir, "models", lewm_filename)
    if not os.path.exists(lewm_path) and is_procgen: # Fallback if name has actual colon
        lewm_path = os.path.join(project_dir, "models", "lewm_vit_procgen_procgen_coinrun_v0.pth")

    print(f"Loading LeWM from: {lewm_path}")
    # Force load dict and ignore missing keys if architecture slightly differs in test script
    state_dict = torch.load(lewm_path, map_location=device)
    lewm.load_state_dict(state_dict, strict=False)
    lewm.eval()

    if is_procgen:
        baseline = PixelPredictor(action_dim=env.action_space.n).to(device)
    else:
        baseline = PixelPredictor(in_channels=1, action_dim=env.action_space.n, out_channels=1).to(device)
    
    base_filename = f"baseline_procgen?procgen_coinrun_v0.pth" if is_procgen else f"baseline_{model_env_id}.pth"
    base_path = os.path.join(project_dir, "models", base_filename)
    print(f"Loading Baseline from: {base_path}")
    baseline.load_state_dict(torch.load(base_path, map_location=device), strict=False)
    baseline.eval()

    print(f"Loaded models for {env_id}")

    lewm_normalizer = OnlineEMANormalizer(alpha=0.99)
    pixel_normalizer = OnlineEMANormalizer(alpha=0.99)

    # 3. Setup Plotting
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    
    # 4. Run Episode
    obs, _ = env.reset(seed=42)
    obs_proc = preprocess_frame(obs)
    
    frame_stack = np.zeros((4, 84, 84), dtype=np.float32)
    def push_frame(f):
        frame_stack[:-1] = frame_stack[1:]
        frame_stack[-1] = f
    
    for _ in range(4):
        push_frame(obs_proc)

    l_hist, p_hist = [], []

    print("Starting visualizer. Press 'q' on the OpenCV window to close it.")

    for step in range(max_steps):
        # Action
        action_val = env.action_space.sample()

        obs_t = torch.tensor(frame_stack, dtype=torch.float32).to(device) / 255.0
        # Thêm batch dimension
        obs_t = obs_t.unsqueeze(0)
        act_t = torch.tensor([action_val], dtype=torch.long).to(device)

        # Trigger OOD
        if step == ood_step:
            if is_procgen:
                env.trigger_ood(momentum_prob=0.3)
            else:
                env.trigger_ood(new_frameskip=new_frameskip)

        next_obs, reward_val, terminated, truncated, _ = env.step(action_val)
        is_id = step < ood_step
        
        next_obs_proc = preprocess_frame(next_obs)
        push_frame(next_obs_proc)
        next_obs_t = torch.tensor(frame_stack, dtype=torch.float32).to(device) / 255.0
        next_obs_t = next_obs_t.unsqueeze(0)

        # Predictions
        with torch.no_grad():
            if is_procgen:
                z_t = lewm.get_latent(obs_t)
                z_t1 = lewm.get_latent(next_obs_t)
            else:
                z_t = lewm.get_latent(obs_t[:, -1:])
                z_t1 = lewm.get_latent(next_obs_t[:, -1:])
                
            pred_z_t1 = lewm.predict_next(z_t, act_t)
            l_surp_raw = float(torch.mean((pred_z_t1 - z_t1) ** 2).item())
            l_norm = lewm_normalizer.update_and_normalize(l_surp_raw, is_id)
            
            if is_procgen:
                pred_obs_t1 = baseline(obs_t, act_t)
                p_surp_raw = float(torch.mean((pred_obs_t1 - next_obs_t) ** 2).item())
            else:
                pred_obs_t1 = baseline(obs_t[:, -1:], act_t)
                p_surp_raw = float(torch.mean((pred_obs_t1 - next_obs_t[:, -1:]) ** 2).item())
            
            p_norm = pixel_normalizer.update_and_normalize(p_surp_raw, is_id)

        l_hist.append(l_norm)
        p_hist.append(p_norm)

        # Rendering
        game_rgb = env.render()
        if game_rgb is None:
            game_rgb = next_obs # fallback
            
        game_bgr = cv2.cvtColor(game_rgb, cv2.COLOR_RGB2BGR)
        game_resized = cv2.resize(game_bgr, (400, 400), interpolation=cv2.INTER_AREA)

        plot_bgr = plot_to_img(fig, ax, l_hist, p_hist, ood_step, step)
        plot_resized = cv2.resize(plot_bgr, (600, 400), interpolation=cv2.INTER_AREA)

        combined = np.hstack((game_resized, plot_resized))
        
        # Add overlay text
        status_text = "OOD ACTIVE (State Complexity Bias)" if not is_id and is_procgen else ("OOD ACTIVE (Physics Fast)" if not is_id else "NORMAL (ID)")
        color = (0, 0, 255) if not is_id else (0, 255, 0)
        cv2.putText(combined, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow(f"LeWM Live - {env_id.upper()}", combined)
        # Delay ~30ms to make it run at ~30 FPS, otherwise it flashes by too fast
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

        if terminated or truncated:
            break

    env.close()
    cv2.destroyAllWindows()
    plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="pong", choices=["pong", "procgen"], help="Environment to visualize")
    args = parser.parse_args()
    run_live_visualization(args.env)
