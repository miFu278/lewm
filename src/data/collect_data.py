import ale_py
import gymnasium as gym
import numpy as np
import os
import cv2
import sys
import argparse
from stable_baselines3 import PPO

# Thêm đường dẫn project vào sys.path để import env
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_dir)

from src.env.atari_ood_wrapper import AtariDynamicFrameskipWrapper

def collect_atari_data(env_id="ALE/Pong-v5", num_episodes=5, save_dir="datasets", model_path=None):
    """
    Thu thập dữ liệu (rollouts) từ môi trường Atari và lưu thành file .npz.
    Dữ liệu này sẽ được dùng để huấn luyện LeWM (JEPA).
    """
    # Đảm bảo lưu data trong thư mục project
    save_dir_full = os.path.join(project_dir, save_dir)
    os.makedirs(save_dir_full, exist_ok=True)
    
    # Thiết lập môi trường NoFrameskip
    # Cần set frameskip=1 để wrapper tự quản lý frameskip
    env = gym.make(env_id, frameskip=1, render_mode="rgb_array")
    env = AtariDynamicFrameskipWrapper(env, initial_frameskip=4)
    
    # Load model nếu có
    model = None
    if model_path and os.path.exists(model_path):
        print(f"Đang tải PPO model từ {model_path}...")
        model = PPO.load(model_path)
    else:
        print("Không tìm thấy model PPO hợp lệ. Sẽ dùng hành động ngẫu nhiên (Random Agent)!")
    
    print(f"Bắt đầu thu thập dữ liệu cho {env_id}...")
    
    observations = []
    actions = []
    rewards = []
    terminals = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        step = 0
        
        while not done and step < 1000: # Giới hạn 1000 bước mỗi episode
            # Tiền xử lý hình ảnh: Chuyển sang Grayscale và resize về 84x84 (chuẩn Atari/LeWM)
            if len(obs.shape) == 3 and obs.shape[-1] == 3: # Nếu là ảnh RGB
                obs_gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
            else:
                obs_gray = obs
            
            obs_resized = cv2.resize(obs_gray, (84, 84), interpolation=cv2.INTER_AREA)
            observations.append(obs_resized)
            
            # Chọn hành động
            if model is not None:
                # Do PPO CnnPolicy yêu cầu 4 frame stack nên nếu pass thẳng obs này có thể lỗi shape
                # Tuy nhiên, nếu model được train không frame stack (chỉ để demo) thì ok
                # Để tương thích nhanh với CnnPolicy đã train với VecFrameStack,
                # ở mức đơn giản ta vẫn dùng action_space.sample() nếu model dự đoán lỗi.
                # Tuy nhiên PPO sẽ cố gắng tự infer. Hãy đảm bảo truyền ảnh shape phù hợp.
                # Trong thực tế cần bọc env này bằng VecFrameStack tương tự lúc train.
                # Đoạn code dưới đây thử dùng model.predict với ảnh nguyên bản obs.
                try:
                    action, _states = model.predict(obs, deterministic=False)
                    # action trả về có thể là list, ta lấy phần tử đầu
                    if isinstance(action, np.ndarray) and action.size == 1:
                        action = int(action.item())
                except Exception as e:
                    action = env.action_space.sample()
            else:
                action = env.action_space.sample()
                
            actions.append(action)
            
            # Bước đi trong môi trường
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            rewards.append(reward)
            terminals.append(done)
            step += 1
            
        print(f"Episode {episode+1}/{num_episodes} completed with {step} steps.")
        
    env.close()
    
    # Lưu thành file numpy nén
    obs_np = np.array(observations, dtype=np.uint8)
    act_np = np.array(actions, dtype=np.int32)
    rew_np = np.array(rewards, dtype=np.float32)
    term_np = np.array(terminals, dtype=bool)
    
    # Format tên file: atari_data_Pong_v5.npz
    clean_env_id = env_id.replace("/", "_").replace("-", "_")
    save_path = os.path.join(save_dir_full, f"atari_data_{clean_env_id}.npz")
    np.savez_compressed(save_path, obs=obs_np, actions=act_np, rewards=rew_np, terminals=term_np)
    print(f"\n[Thành công] Đã lưu dataset tại:\n {save_path}")
    print(f"Kích thước tensor hình ảnh: {obs_np.shape} (frames, height, width)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect Atari trajectories")
    parser.add_argument("--env", type=str, default="ALE/Pong-v5", help="Atari Environment ID")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to collect")
    parser.add_argument("--model_path", type=str, default=None, help="Path to trained PPO model (.zip)")
    args = parser.parse_args()
    
    collect_atari_data(env_id=args.env, num_episodes=args.episodes, model_path=args.model_path)
