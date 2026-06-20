import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import argparse
import sys
from tqdm import tqdm
import copy

# Thêm đường dẫn project vào sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
if project_dir not in sys.path:
    sys.path.append(project_dir)

from src.models.lewm import LeWorldModel, SIGReg
from src.models.baseline import PixelPredictor

class TransitionDataset(Dataset):
    """
    Dataset cho dữ liệu chuyển dịch (transitions): (s_t, a_t, s_{t+1})
    """
    def __init__(self, data_paths):
        if isinstance(data_paths, str):
            data_paths = [data_paths]
            
        all_obs = []
        all_actions = []
        all_terminals = []
        
        for data_path in data_paths:
            if not os.path.exists(data_path):
                print(f"Bỏ qua file không tồn tại: {data_path}")
                continue
                
            print(f"Đang tải dataset từ {data_path}...")
            data = np.load(data_path)
            all_obs.append(data['obs'])
            all_actions.append(data['actions'])
            all_terminals.append(data['terminals'])
            
        if not all_obs:
            raise FileNotFoundError("Không có dataset nào hợp lệ để tải!")
            
        self.obs = np.concatenate(all_obs, axis=0) # shape [N, 84, 84]
        self.actions = np.concatenate(all_actions, axis=0) # shape [N]
        self.terminals = np.concatenate(all_terminals, axis=0) # shape [N]
        
        # Tạo mảng lưu vị trí bắt đầu của mỗi episode để build frame stack
        self.start_idx = np.zeros(len(self.obs), dtype=int)
        curr_start = 0
        for i in range(len(self.obs)):
            self.start_idx[i] = curr_start
            if self.terminals[i]:
                curr_start = i + 1
                
        # Tạo danh sách các index chuyển dịch hợp lệ (không bị ngắt bởi terminal/done)
        self.valid_indices = []
        for i in range(len(self.obs) - 1):
            if not self.terminals[i]:
                self.valid_indices.append(i)
                
        print(f"Tổng số transitions hợp lệ: {len(self.valid_indices)} trên {len(self.obs)} frames (sau khi gộp).")

    def __len__(self):
        return len(self.valid_indices)

    def get_stack(self, end_idx):
        start_i = self.start_idx[end_idx]
        stack = []
        for j in range(end_idx - 3, end_idx + 1):
            j_valid = max(j, start_i)
            stack.append(self.obs[j_valid])
        return np.stack(stack, axis=0)

    def __getitem__(self, idx):
        i = self.valid_indices[idx]
        # Frame stack [4, 84, 84] chuẩn hóa về [0, 1]
        obs_t = torch.tensor(self.get_stack(i), dtype=torch.float32) / 255.0
        action_t = torch.tensor(self.actions[i], dtype=torch.long)
        obs_t1 = torch.tensor(self.get_stack(i+1), dtype=torch.float32) / 255.0
        return obs_t, action_t, obs_t1

def train_models(env_id="ALE/Pong-v5", epochs=15, batch_size=64, lr=1e-3, lambda_sig=0.1, latent_dim=64):
    import glob
    clean_env_id = env_id.replace("/", "_").replace("-", "_")
    search_pattern = os.path.join(project_dir, "datasets", f"atari_data_{clean_env_id}*.npz")
    dataset_files = glob.glob(search_pattern)
    
    if not dataset_files:
        default_path = os.path.join(project_dir, "datasets", f"atari_data_{clean_env_id}.npz")
        dataset_files = [default_path]
    
    # 1. Khởi tạo Dataset & DataLoader
    try:
        # Load và gộp toàn bộ các file dataset tìm được
        dataset = TransitionDataset(dataset_files)
    except FileNotFoundError as e:
        print(f"Lỗi: {e}")
        print("Vui lòng chạy collect_data.py trước để tạo dataset!")
        return
        
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    # Xác định số hành động từ môi trường
    # Pong có 6 actions. Hãy lấy thông tin này tự động nếu có thể, hoặc mặc định 6
    action_dim = 6
    if "procgen" in env_id.lower():
        action_dim = 15
    elif "Breakout" in env_id:
        action_dim = 4
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Huấn luyện sử dụng thiết bị: {device}")
    
    # Đường dẫn lưu models
    models_dir = os.path.join(project_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    lewm_save_path = os.path.join(models_dir, f"lewm_vit_{clean_env_id}.pth")
    baseline_save_path = os.path.join(models_dir, f"baseline_{clean_env_id}.pth")

    # 2. Khởi tạo LeWorldModel
    print("\n--- Khởi tạo LeWorldModel (JEPA) ---")
    lewm = LeWorldModel(action_dim=action_dim, embed_dim=192).to(device)
    if os.path.exists(lewm_save_path):
        print(f"Tìm thấy trọng số LeWM cũ tại {lewm_save_path}. Đang tải lên để train tiếp...")
        lewm.load_state_dict(torch.load(lewm_save_path, map_location=device))
    optimizer_lewm = optim.Adam(lewm.parameters(), lr=lr)
    
    # Khởi tạo Target Encoder (bản sao của lewm.encoder)
    target_encoder = copy.deepcopy(lewm.encoder).to(device)
    for p in target_encoder.parameters():
        p.requires_grad = False
    
    # 3. Khởi tạo Baseline (Pixel Predictor)
    print("--- Khởi tạo Pixel Reconstruction Baseline ---")
    baseline = PixelPredictor(action_dim=action_dim).to(device)
    if os.path.exists(baseline_save_path):
        print(f"Tìm thấy trọng số Baseline cũ tại {baseline_save_path}. Đang tải lên để train tiếp...")
        baseline.load_state_dict(torch.load(baseline_save_path, map_location=device))
    optimizer_baseline = optim.Adam(baseline.parameters(), lr=lr)
    
    # 4. Huấn luyện
    print(f"\nBắt đầu huấn luyện trong {epochs} epochs...")
    sigreg_module = SIGReg(num_proj=1024).to(device)
    
    for epoch in range(epochs):
        lewm.train()
        baseline.train()
        
        epoch_pred_loss = 0.0
        epoch_sig_loss = 0.0
        epoch_lewm_total_loss = 0.0
        epoch_base_loss = 0.0
        
        for batch_idx, (obs_t, action_t, obs_t1) in enumerate(tqdm(dataloader, desc=f"Epoch [{epoch+1}/{epochs}]", leave=False)):
            obs_t = obs_t.to(device)
            action_t = action_t.to(device)
            obs_t1 = obs_t1.to(device)
            
            # --- Huấn luyện LeWorldModel ---
            optimizer_lewm.zero_grad()
            
            # Forward Online
            z_t = lewm.get_latent(obs_t)
            pred_z_t1 = lewm.predict_next(z_t, action_t)
            
            # Forward Target (Không có gradient)
            with torch.no_grad():
                z_t1_target = target_encoder(obs_t1)
            
            # Loss dự đoán
            pred_loss = F_mse_loss = nn.MSELoss()(pred_z_t1, z_t1_target)
            
            # Regularize các vector tạo ra bởi mạng Online bằng SIGReg
            z_all = torch.cat([z_t, pred_z_t1], dim=0)
            sig_loss = sigreg_module(z_all.unsqueeze(0))
            
            total_lewm_loss = pred_loss + lambda_sig * sig_loss
            
            total_lewm_loss.backward()
            optimizer_lewm.step()
            
            # Cập nhật Target Encoder (EMA)
            ema_tau = 0.99
            for p_target, p_online in zip(target_encoder.parameters(), lewm.encoder.parameters()):
                p_target.data.mul_(ema_tau).add_((1 - ema_tau) * p_online.data)
            
            epoch_pred_loss += pred_loss.item()
            epoch_sig_loss += sig_loss.item()
            epoch_lewm_total_loss += total_lewm_loss.item()
            
            # --- Huấn luyện Baseline (Pixel Predictor) ---
            optimizer_baseline.zero_grad()
            
            pred_obs_t1 = baseline(obs_t, action_t)
            base_loss = nn.MSELoss()(pred_obs_t1, obs_t1)
            
            base_loss.backward()
            optimizer_baseline.step()
            
            epoch_base_loss += base_loss.item()
            
        num_batches = len(dataloader)
        avg_pred = epoch_pred_loss / num_batches
        avg_sig = epoch_sig_loss / num_batches
        avg_lewm_total = epoch_lewm_total_loss / num_batches
        avg_base = epoch_base_loss / num_batches
        
        print(f"Epoch [{epoch+1}/{epochs}] | "
              f"LeWM Total Loss: {avg_lewm_total:.6f} (Pred: {avg_pred:.6f}, SIG: {avg_sig:.6f}) | "
              f"Baseline Pixel Loss: {avg_base:.6f}")
              
    # 5. Lưu models
    torch.save(lewm.state_dict(), lewm_save_path)
    torch.save(baseline.state_dict(), baseline_save_path)
    
    print(f"\n[Thành công] Đã lưu models:")
    print(f" - LeWorldModel: {lewm_save_path}")
    print(f" - Baseline Pixel: {baseline_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LeWorldModel and Reconstruction Baseline")
    parser.add_argument("--env", type=str, default="ALE/Pong-v5", help="Atari Environment ID")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--lambda_sig", type=float, default=0.1, help="SIGReg loss scaling factor")
    parser.add_argument("--latent_dim", type=int, default=64, help="Latent space dimension for LeWM")
    args = parser.parse_args()
    
    train_models(
        env_id=args.env, 
        epochs=args.epochs, 
        batch_size=args.batch_size, 
        lr=args.lr, 
        lambda_sig=args.lambda_sig,
        latent_dim=args.latent_dim
    )
