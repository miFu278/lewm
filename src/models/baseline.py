import torch
import torch.nn as nn
import torch.nn.functional as F

class PixelPredictor(nn.Module):
    """
    Encoder-Decoder CNN that predicts next 84x84 frame s_{t+1} given current s_t and action a_t.
    """
    def __init__(self, action_dim=6, action_embed_dim=8):
        super().__init__()
        # Encoder: 84x84 -> 42x42 -> 21x21
        self.enc_conv1 = nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1)
        self.enc_conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1)
        
        # Action Embedding
        self.action_embedding = nn.Embedding(action_dim, action_embed_dim)
        
        # Decoder: 21x21 -> 42x42 -> 84x84
        # Input channel count to decoder is 32 (latent feature map) + action_embed_dim
        self.dec_conv1 = nn.ConvTranspose2d(32 + action_embed_dim, 16, kernel_size=4, stride=2, padding=1)
        self.dec_conv2 = nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1)

    def forward(self, obs, action):
        # obs: [B, 1, 84, 84] in [0, 1] range
        # action: [B]
        
        # Encode state
        h = F.relu(self.enc_conv1(obs)) # [B, 16, 42, 42]
        h = F.relu(self.enc_conv2(h))   # [B, 32, 21, 21]
        
        # Embed action and broadcast spatially
        act_emb = self.action_embedding(action) # [B, action_embed_dim]
        act_emb_spatial = act_emb.view(act_emb.size(0), act_emb.size(1), 1, 1)
        act_emb_spatial = act_emb_spatial.expand(-1, -1, h.size(2), h.size(3)) # [B, action_embed_dim, 21, 21]
        
        # Concatenate features and action representation
        h_combined = torch.cat([h, act_emb_spatial], dim=1) # [B, 32 + action_embed_dim, 21, 21]
        
        # Decode next state
        h_dec = F.relu(self.dec_conv1(h_combined)) # [B, 16, 42, 42]
        pred_next_obs = torch.sigmoid(self.dec_conv2(h_dec)) # [B, 1, 84, 84]
        
        return pred_next_obs
