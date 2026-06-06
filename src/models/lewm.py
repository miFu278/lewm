import torch
import torch.nn as nn
import torch.nn.functional as F

def sigreg_loss(x, sketch_dim=64):
    """
    Sliced Isotropic Gaussian Regularizer (SIGReg) Loss.
    Enforces the latent representations to follow a standard normal distribution.
    Uses Euler's formula representation to handle complex exponentials robustly.
    """
    N, C = x.size()
    if N <= 1:
        return torch.tensor(0.0, device=x.device, requires_grad=True)
        
    # 1. Random Projection (The Observer)
    # Project channels down to sketch_dim
    A = torch.randn(C, sketch_dim, device=x.device)
    A = A / (A.norm(p=2, dim=0, keepdim=True) + 1e-6)
    
    # 2. Integration Points
    t = torch.linspace(-5, 5, 17, device=x.device)
    
    # 3. Theoretical Gaussian Characteristic Function (CF)
    exp_f = torch.exp(-0.5 * t**2) # [T]
    
    # 4. Empirical CF using Euler's formula: exp(i*theta) = cos(theta) + i*sin(theta)
    proj = x @ A  # [N, sketch_dim]
    args = proj.unsqueeze(2) * t.view(1, 1, -1) # [N, sketch_dim, T]
    
    ecf_real = torch.cos(args).mean(dim=0) # [sketch_dim, T]
    ecf_imag = torch.sin(args).mean(dim=0) # [sketch_dim, T]
    
    # 5. Weighted L2 Distance between Empirical and Theoretical CF
    diff_sq = (ecf_real - exp_f.unsqueeze(0)).square() + ecf_imag.square() # [sketch_dim, T]
    err = diff_sq * exp_f.unsqueeze(0) # [sketch_dim, T]
    
    # 6. Integrate using trapezoid rule
    if hasattr(torch, 'trapezoid'):
        loss = torch.trapezoid(err, t, dim=1) * N
    else:
        # Fallback for older PyTorch versions
        loss = torch.trapz(err, t, dim=1) * N
        
    return loss.mean()

class AtariEncoder(nn.Module):
    """
    Lightweight CNN Encoder mapping 84x84 grayscale frames to a latent space.
    """
    def __init__(self, latent_dim=64):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc = nn.Linear(64 * 7 * 7, latent_dim)

    def forward(self, x):
        # Input should be [B, 1, 84, 84] in [0, 1] range
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x

class LatentPredictor(nn.Module):
    """
    MLP Predictor that predicts next latent state z_{t+1} given current latent state z_t and action a_t.
    """
    def __init__(self, latent_dim=64, action_dim=6, action_embed_dim=16):
        super().__init__()
        self.action_embedding = nn.Embedding(action_dim, action_embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim + action_embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )

    def forward(self, z, action):
        # z: [B, latent_dim]
        # action: [B]
        act_emb = self.action_embedding(action) # [B, action_embed_dim]
        inputs = torch.cat([z, act_emb], dim=1) # [B, latent_dim + action_embed_dim]
        pred_z = self.mlp(inputs)
        return pred_z

class LeWorldModel(nn.Module):
    """
    LeWorldModel (JEPA): Shared weight encoder for context and target states,
    with a latent transition predictor, regularized by SIGReg loss to prevent collapse.
    """
    def __init__(self, latent_dim=64, action_dim=6):
        super().__init__()
        self.encoder = AtariEncoder(latent_dim)
        self.predictor = LatentPredictor(latent_dim, action_dim)

    def get_latent(self, x):
        return self.encoder(x)

    def predict_next(self, z, action):
        return self.predictor(z, action)

    def forward(self, obs, action):
        # Utility forward pass
        z = self.get_latent(obs)
        pred_z = self.predict_next(z, action)
        return pred_z
