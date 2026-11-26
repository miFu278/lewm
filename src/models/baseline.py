import torch
import torch.nn as nn
import torch.nn.functional as F
 
 
class PixelPredictor(nn.Module):
    """
    CNN Encoder-Decoder dự đoán pixel frame kế tiếp.
    Action được broadcast không gian (spatial) thay vì flatten,
    giúp model học được spatial correlation của action.
 
    Flow:
        obs [B,4,84,84]
          → enc_conv1 (k4,s2,p1) → [B,16,42,42]
          → enc_conv2 (k4,s2,p1) → [B,32,21,21]
        action [B] → Embedding → [B,8] → broadcast → [B,8,21,21]
        concat                              → [B,40,21,21]
          → dec_conv1 (k4,s2,p1) → [B,16,42,42]
          → dec_conv2 (k4,s2,p1) + Sigmoid → [B,4,84,84] ∈ [0,1]
 
    Args:
        action_dim:       Số lượng action rời rạc (Pong=6, Breakout=4).
        action_embed_dim: Chiều embedding action (default=8, nhỏ → tiết kiệm VRAM).
    """
 
    def __init__(self, in_channels: int = 4, action_dim: int = 6, action_embed_dim: int = 8, out_channels: int = 4):
        super().__init__()
 
        # --- Encoder ---
        self.enc_conv1 = nn.Conv2d(in_channels, 16, kernel_size=4, stride=2, padding=1)
        self.enc_conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1)
 
        # Action Embedding
        self.action_embedding = nn.Embedding(action_dim, action_embed_dim)
 
        # --- Decoder ---
        # Feature map size sau enc_conv2 lA 21x21 (vA 84 -> 42 -> 21)
        # ChAng ta sA broadcast action embedding thAnh 21x21 vA concat vAo cA1ng feature map
        self.dec_conv1 = nn.ConvTranspose2d(32 + action_embed_dim, 16, kernel_size=4, stride=2, padding=1)
        self.dec_conv2 = nn.ConvTranspose2d(16, out_channels, kernel_size=4, stride=2, padding=1)
 
    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs:    [B, 4, 84, 84] float32 trong [0, 1]
            action: [B] long
 
        Returns:
            pred_next_obs: [B, 4, 84, 84] float32 trong [0, 1]
        """
        # Encode
        h = F.relu(self.enc_conv1(obs))   # [B, 16, 42, 42]
        h = F.relu(self.enc_conv2(h))     # [B, 32, 21, 21]
 
        # Action: embed → broadcast spatial
        act_emb = self.action_embedding(action)                       # [B, 8]
        act_spatial = act_emb.view(act_emb.size(0), act_emb.size(1), 1, 1)
        act_spatial = act_spatial.expand(-1, -1, h.size(2), h.size(3))  # [B, 8, 21, 21]
 
        # Fuse
        h = torch.cat([h, act_spatial], dim=1)  # [B, 40, 21, 21]
 
        # Decode
        h = F.relu(self.dec_conv1(h))            # [B, 16, 42, 42]
        pred_next_obs = torch.sigmoid(self.dec_conv2(h))  # [B, 4, 84, 84]
 
        return pred_next_obs
