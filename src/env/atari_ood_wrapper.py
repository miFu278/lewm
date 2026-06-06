import gymnasium as gym
import numpy as np

class AtariDynamicFrameskipWrapper(gym.Wrapper):
    """
    Wrapper tùy chỉnh cho môi trường Atari NoFrameskip.
    Cho phép thay đổi frameskip động (đột ngột) giữa chừng để giả lập OOD Dynamics.
    Bao gồm cả tính năng max-pooling 2 frame cuối (tiêu chuẩn của Atari).
    """
    def __init__(self, env, initial_frameskip=4):
        super().__init__(env)
        self.frameskip = initial_frameskip
        self.obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)

    def trigger_ood(self, new_frameskip):
        """
        Kích hoạt sự thay đổi động học.
        Ví dụ: từ frameskip=4 chuyển sang frameskip=2 (game chậm lại một nửa)
        """
        print(f"[OOD Triggered] Changing frameskip from {self.frameskip} to {new_frameskip}")
        self.frameskip = new_frameskip

    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        
        obs = None
        for i in range(self.frameskip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            
            if self.frameskip > 1:
                if i == self.frameskip - 2:
                    self.obs_buffer[0] = obs
                if i == self.frameskip - 1:
                    self.obs_buffer[1] = obs
                
            if terminated or truncated:
                break
                
        # Max pooling over last 2 frames for Atari to remove flickering
        if self.frameskip >= 2:
            max_frame = self.obs_buffer.max(axis=0)
        else:
            max_frame = obs
            
        return max_frame, total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.obs_buffer[0] = obs
        self.obs_buffer[1] = obs
        return obs, info

if __name__ == "__main__":
    # Test script nhanh
    env = gym.make("ALE/Pong-v5", frameskip=1, render_mode="human") # Cần frameskip=1 (hoặc dùng PongNoFrameskip-v4)
    env = AtariDynamicFrameskipWrapper(env, initial_frameskip=4)
    
    obs, info = env.reset()
    for step in range(500):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        
        # Ở step 200, kích hoạt OOD
        if step == 200:
            env.trigger_ood(new_frameskip=1) # Đột nhiên siêu chậm
            
        if term or trunc:
            obs, info = env.reset()
            
    env.close()
