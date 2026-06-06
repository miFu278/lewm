import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import argparse
import sys

# Thêm đường dẫn project vào sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
if project_dir not in sys.path:
    sys.path.append(project_dir)

from src.models.lewm import LeWorldModel, sigreg_loss
from src.models.baseline import PixelPredictor

class TransitionDataset(Dataset):
    """
    Dataset cho dữ liệu chuyển dịch (transitions): (s_t, a_t, s_{t+1})
    """
    def __init__(self, data_path):
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Không tìm thấy file dataset tại: {data_path}")
            
        print(f"Đang tải dataset từ {data_path}...")
        data = np.load(data_path)
        self.obs = data['obs'] # shape [N, 84, 84]
        self.actions = data['actions'] # shape [N]
        self.terminals = data['terminals'] # shape [N]
        
        # Tạo danh sách các index chuyển dịch hợp lệ (không bị ngắt bởi terminal/done)
        self.valid_indices = []
        for i in range(len(self.obs) - 1):
            if not self.terminals[i]:
                self.valid_indices.append(i)
                
        print(f"Tổng số transitions hợp lệ: {len(self.valid_indices)} trên {len(self.obs)} frames.")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        i = self.valid_indices[idx]
        # Thêm channel dimension [1, 84, 84] và chuẩn hóa về [0, 1]
        obs_t = torch.tensor(self.obs[i], dtype=torch.float32).unsqueeze(0) / 255.0
        action_t = torch.tensor(self.actions[i], dtype=torch.long)
        obs_t1 = torch.tensor(self.obs[i+1], dtype=torch.float32).unsqueeze(0) / 255.0
        return obs_t, action_t, obs_t1

def train_models(env_id="ALE/Pong-v5", epochs=15, batch_size=64, lr=1e-3, lambda_sig=0.1, latent_dim=64):
    clean_env_id = env_id.replace("/", "_").replace("-", "_")
    dataset_path = os.path.join(project_dir, "datasets", f"atari_data_{clean_env_id}.npz")
    
    # 1. Khởi tạo Dataset & DataLoader
    try:
        dataset = TransitionDataset(dataset_path)
    except FileNotFoundError as e:
        print(f"Lỗi: {e}")
        print("Vui lòng chạy collect_data.py trước để tạo dataset!")
        return
        
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    # Xác định số hành động từ môi trường
    # Pong có 6 actions. Hãy lấy thông tin này tự động nếu có thể, hoặc mặc định 6
    action_dim = 6
    if "Breakout" in env_id:
        action_dim = 4
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Huấn luyện sử dụng thiết bị: {device}")
    
    # 2. Khởi tạo LeWorldModel
    print("\n--- Khởi tạo LeWorldModel (JEPA) ---")
    lewm = LeWorldModel(latent_dim=latent_dim, action_dim=action_dim).to(device)
    optimizer_lewm = optim.Adam(lewm.parameters(), lr=lr)
    
    # 3. Khởi tạo Baseline (Pixel Predictor)
    print("--- Khởi tạo Pixel Reconstruction Baseline ---")
    baseline = PixelPredictor(action_dim=action_dim).to(device)
    optimizer_baseline = optim.Adam(baseline.parameters(), lr=lr)
    
    # 4. Huấn luyện
    print(f"\nBắt đầu huấn luyện trong {epochs} epochs...")
    
    for epoch in range(epochs):
        lewm.train()
        baseline.train()
        
        epoch_pred_loss = 0.0
        epoch_sig_loss = 0.0
        epoch_lewm_total_loss = 0.0
        epoch_base_loss = 0.0
        
        for batch_idx, (obs_t, action_t, obs_t1) in enumerate(dataloader):
            obs_t = obs_t.to(device)
            action_t = action_t.to(device)
            obs_t1 = obs_t1.to(device)
            
            # --- Huấn luyện LeWorldModel ---
            optimizer_lewm.zero_grad()
            
            # Forward
            z_t = lewm.get_latent(obs_t)
            z_t1 = lewm.get_latent(obs_t1)
            pred_z_t1 = lewm.predict_next(z_t, action_t)
            
            # Loss
            pred_loss = F_mse_loss = nn.MSELoss()(pred_z_t1, z_t1)
            # Regularize cả z_t và z_t1 bằng SIGReg
            z_all = torch.cat([z_t, z_t1], dim=0)
            sig_loss = sigreg_loss(z_all)
            
            total_lewm_loss = pred_loss + lambda_sig * sig_loss
            
            total_lewm_loss.backward()
            optimizer_lewm.step()
            
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
    models_dir = os.path.join(project_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    lewm_save_path = os.path.join(models_dir, f"lewm_{clean_env_id}.pth")
    baseline_save_path = os.path.join(models_dir, f"baseline_{clean_env_id}.pth")
    
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
