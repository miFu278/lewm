import ale_py
import gymnasium as gym
import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env, make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback

def train_ppo_atari(env_id="ALE/Pong-v5", total_timesteps=100000, save_dir="models"):
    """
    Huấn luyện mô hình PPO cơ bản trên môi trường Atari.
    """
    # Đảm bảo thư mục lưu trữ tồn tại
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(current_dir)
    save_dir_full = os.path.join(project_dir, save_dir)
    os.makedirs(save_dir_full, exist_ok=True)
    
    print(f"Khởi tạo môi trường {env_id}...")
    # Nếu là Procgen, ta dùng make_vec_env
    if "procgen" in env_id.lower():
        vec_env = make_vec_env(env_id, n_envs=4, seed=0)
    else:
        # Sử dụng make_atari_env để tạo môi trường chuẩn Atari
        vec_env = make_atari_env(env_id, n_envs=4, seed=0)
        
    # Xếp chồng 4 frame liên tiếp để giúp mạng học được motion (dùng chung để đồng nhất shape đầu vào)
    vec_env = VecFrameStack(vec_env, n_stack=4)
    
    # Format tên file
    clean_env_id = env_id.replace("/", "_").replace("-", "_")
    save_path = os.path.join(save_dir_full, f"ppo_{clean_env_id}")
    
    # Kiểm tra xem có model cũ không để học tiếp
    model_file = save_path + ".zip"
    if os.path.exists(model_file):
        print(f"Tìm thấy model cũ tại {model_file}. Đang tải lên để học tiếp (Resume Training)...")
        # Load model cũ và set env mới vào để tiếp tục train
        model = PPO.load(model_file, env=vec_env, tensorboard_log=os.path.join(project_dir, "logs"))
    else:
        print(f"Khởi tạo PPO model mới...")
        # Dùng CnnPolicy cho đầu vào dạng hình ảnh
        model = PPO("CnnPolicy", vec_env, verbose=1, tensorboard_log=os.path.join(project_dir, "logs"))
    
    # Tự động lưu checkpoint sau mỗi 20.000 timesteps (save_freq = 20000 / n_envs)
    checkpoint_callback = CheckpointCallback(
        save_freq=max(1000, 20000 // 4),
        save_path=save_dir_full,
        name_prefix=f"ppo_{clean_env_id}"
    )
    
    print(f"Bắt đầu huấn luyện với {total_timesteps} steps...")
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
    
    print(f"Lưu model cuối cùng tại {save_path}.zip")
    model.save(save_path)
    
    return save_path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train PPO Agent on Atari")
    parser.add_argument("--env", type=str, default="ALE/Pong-v5", help="Atari Environment ID")
    parser.add_argument("--timesteps", type=int, default=100000, help="Total training timesteps")
    args = parser.parse_args()
    
    train_ppo_atari(env_id=args.env, total_timesteps=args.timesteps)
