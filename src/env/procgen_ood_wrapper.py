import gymnasium as gym
import numpy as np

class ProcgenOODWrapper(gym.Wrapper):
    """
    Wrapper cho môi trường Procgen để giả lập OOD Dynamics.
    Do Procgen dùng C++ engine khó can thiệp trực tiếp physics, ta sẽ mô phỏng
    bằng cách thêm quán tính (momentum) ngẫu nhiên vào hành động của Agent.
    """
    def __init__(self, env):
        super().__init__(env)
        self.is_ood = False
        self.last_action = None
        self.momentum_prob = 0.0

    def trigger_ood(self, momentum_prob=0.3):
        """
        Kích hoạt OOD. momentum_prob là xác suất Agent bị trượt 
        (lặp lại hành động trước đó dù đã bấm nút khác).
        """
        print(f"[OOD Triggered] Adding action momentum with probability {momentum_prob}")
        self.is_ood = True
        self.momentum_prob = momentum_prob

    def step(self, action):
        if self.is_ood:
            # Mô phỏng quán tính (trượt): có xác suất lặp lại hành động cũ
            if self.last_action is not None and np.random.rand() < self.momentum_prob:
                action_to_take = self.last_action
            else:
                action_to_take = action
        else:
            action_to_take = action

        self.last_action = action_to_take
        return self.env.step(action_to_take)

    def reset(self, **kwargs):
        self.last_action = None
        return self.env.reset(**kwargs)
