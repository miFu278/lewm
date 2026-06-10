import gymnasium as gym
import numpy as np
 
 
class AtariDynamicFrameskipWrapper(gym.Wrapper):
    """
    Wrapper cho môi trường Atari (tạo với frameskip=1).
    Thực hiện frameskip thủ công + max-pooling 2 frame cuối,
    và hỗ trợ chuyển đổi OOD dynamics động giữa chừng.
 
    Args:
        env: Môi trường Atari tạo với frameskip=1.
        initial_frameskip: Frameskip In-Distribution (default=4).
 
    Cách dùng:
        env = gym.make("ALE/Pong-v5", frameskip=1, render_mode="rgb_array")
        env = AtariDynamicFrameskipWrapper(env, initial_frameskip=4)
        obs, _ = env.reset()
 
        # Training/ID phase ...
        env.trigger_ood(new_frameskip=2)   # Kích hoạt OOD
        # OOD phase ...
        env.reset_to_id()                  # Trở về ID (cho ablation)
    """
 
    def __init__(self, env: gym.Env, initial_frameskip: int = 4):
        super().__init__(env)
        assert initial_frameskip >= 1, "initial_frameskip phải >= 1"
        self.initial_frameskip = initial_frameskip
        self.frameskip = initial_frameskip
        self.is_ood = False
 
        # Buffer 2 frame cuối để max-pool (loại flickering chuẩn Atari)
        self.obs_buffer = np.zeros(
            (2,) + env.observation_space.shape, dtype=np.uint8
        )
 
    # ── Public API ────────────────────────────────────────────────────────────
 
    def trigger_ood(self, new_frameskip: int):
        """Kích hoạt OOD dynamics bằng cách đổi frameskip."""
        assert new_frameskip >= 1, "new_frameskip phải >= 1"
        print(f"[OOD Triggered] frameskip: {self.frameskip} → {new_frameskip}")
        self.frameskip = new_frameskip
        self.is_ood = True
 
    def reset_to_id(self):
        """Khôi phục In-Distribution dynamics (dùng trong ablation)."""
        print(f"[ID Restored] frameskip: {self.frameskip} → {self.initial_frameskip}")
        self.frameskip = self.initial_frameskip
        self.is_ood = False
 
    # ── Core logic ────────────────────────────────────────────────────────────
 
    def step(self, action):
        """
        Thực hiện `self.frameskip` bước với cùng action.
        Áp dụng max-pooling 2 frame cuối để loại flickering.
 
        Returns: (max_pooled_obs, total_reward, terminated, truncated, info)
        """
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}
 
        for i in range(self.frameskip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
 
            # Luôn cập nhật buffer — FIX BUG: điều kiện gốc bỏ sót frameskip=1
            if i == max(0, self.frameskip - 2):
                self.obs_buffer[0] = obs
            if i == self.frameskip - 1:
                self.obs_buffer[1] = obs
 
            if terminated or truncated:
                # Đảm bảo buffer[1] luôn có frame cuối cùng khi early-exit
                self.obs_buffer[1] = obs
                break
 
        # Max-pool để loại flickering artifact chuẩn Atari
        max_frame = self.obs_buffer.max(axis=0)
        return max_frame, total_reward, terminated, truncated, info
 
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Khởi tạo cả 2 slot buffer bằng obs đầu tiên
        self.obs_buffer[0] = obs
        self.obs_buffer[1] = obs
        return obs, info
 
 
# ── Quick test ────────────────────────────────────────────────────────────────
 
if __name__ == "__main__":
    import sys
 
    env = gym.make("ALE/Pong-v5", frameskip=1, render_mode="rgb_array")
    env = AtariDynamicFrameskipWrapper(env, initial_frameskip=4)
 
    obs, info = env.reset()
    print(f"Obs shape: {obs.shape}, dtype: {obs.dtype}")
    print(f"Initial frameskip: {env.frameskip}, is_ood: {env.is_ood}")
 
    for step_i in range(300):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
 
        if step_i == 150:
            env.trigger_ood(new_frameskip=2)
            print(f"  Step {step_i}: is_ood={env.is_ood}, frameskip={env.frameskip}")
 
        if step_i == 200:
            env.reset_to_id()
            print(f"  Step {step_i}: is_ood={env.is_ood}, frameskip={env.frameskip}")
 
        if term or trunc:
            obs, info = env.reset()
 
    env.close()
    print("Test passed!")
