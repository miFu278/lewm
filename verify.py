import torch
from src.models.lewm import LeWorldModel

def test():
    print("Khởi tạo mô hình LeWorldModel (ViT + Transformer)...")
    model = LeWorldModel()
    
    print("Tạo dữ liệu ngẫu nhiên (batch_size=2)...")
    obs = torch.randn(2, 1, 84, 84)
    action = torch.randint(0, 15, (2,))
    
    print("Chạy qua Encoder...")
    z = model.get_latent(obs)
    print("Latent shape:", z.shape)
    
    print("Chạy qua Predictor...")
    pred_z = model.predict_next(z, action)
    print("Pred shape:", pred_z.shape)
    
    print("Hoàn tất kiểm tra kiến trúc!")

if __name__ == '__main__':
    test()
