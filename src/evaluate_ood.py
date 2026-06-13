import argparse
import json
import os
import sys
import warnings
warnings.filterwarnings("ignore")
 
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
 
# ── Đường dẫn project ────────────────────────────────────────────────────────
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
if project_dir not in sys.path:
    sys.path.append(project_dir)
 
from src.env.atari_ood_wrapper import AtariDynamicFrameskipWrapper
from src.env.procgen_ood_wrapper import ProcgenOODWrapper
from src.models.lewm import LeWorldModel
from src.models.baseline import PixelPredictor
 
try:
    import gymnasium as gym
    import ale_py  # noqa: F401
except ImportError as e:
    raise ImportError(f"Thiếu dependency: {e}. Chạy: pip install gymnasium ale-py shimmy") from e
 
try:
    from stable_baselines3 import PPO
    HAS_PPO = True
except ImportError:
    HAS_PPO = False
 
 
# ═════════════════════════════════════════════════════════════════════════════
# METRIC FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════
 
def compute_auroc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """
    AUROC theo Wilcoxon-Mann-Whitney (không phụ thuộc sklearn).
    Cao hơn = tốt hơn. Random = 0.5.
    """
    pos = y_scores[y_true == 1]
    neg = y_scores[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    pos_sorted = np.sort(pos)
    idx = np.searchsorted(pos_sorted, neg)
    return float(np.sum(len(pos_sorted) - idx) / (len(pos_sorted) * len(neg)))
 
 
def compute_max_f1(y_true: np.ndarray, y_scores: np.ndarray, n_thresholds: int = 200) -> tuple[float, float]:
    """
    Tính Max F1 và ngưỡng tương ứng qua grid search.
 
    Returns:
        (best_f1, best_threshold)
    """
    thresholds = np.linspace(y_scores.min(), y_scores.max(), n_thresholds)
    best_f1, best_thresh = 0.0, thresholds[0]
    for t in thresholds:
        preds = (y_scores >= t).astype(int)
        tp = np.sum((preds == 1) & (y_true == 1))
        fp = np.sum((preds == 1) & (y_true == 0))
        fn = np.sum((preds == 0) & (y_true == 1))
        if (tp + fp) == 0 or (tp + fn) == 0:
            continue
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        if (p + r) == 0:
            continue
        f1 = 2 * p * r / (p + r)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t
    return float(best_f1), float(best_thresh)
 
 
def compute_ece(y_true: np.ndarray, y_scores: np.ndarray, n_bins: int = 10) -> float:
    """
    Expected Calibration Error (ECE).
 
    Chia y_scores thành n_bins bucket đều nhau trong [0,1] sau khi min-max normalize.
    ECE = Σ |bin_accuracy - bin_confidence| * (bin_size / total)
 
    Thấp hơn = tốt hơn. ECE=0 là perfect calibration.
    """
    # Min-max normalize scores về [0, 1] để interpret như xác suất
    s_min, s_max = y_scores.min(), y_scores.max()
    if s_max == s_min:
        return 0.0
    probs = (y_scores - s_min) / (s_max - s_min)
 
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = len(y_true)
 
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        # Include right edge ở bin cuối
        mask = (probs >= lo) & (probs < hi) if i < n_bins - 1 else (probs >= lo) & (probs <= hi)
        if mask.sum() == 0:
            continue
        bin_conf = probs[mask].mean()
        bin_acc = y_true[mask].mean()
        ece += (mask.sum() / total) * abs(bin_acc - bin_conf)
 
    return float(ece)
 
 
def compute_detection_delay(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    ood_step: int,
    patience: int = 3,
) -> int:
    """
    Detection Delay: số bước sau ood_step cho đến khi detector báo OOD
    liên tục trong `patience` bước.
 
    Returns:
        delay (int): -1 nếu không bao giờ detect được.
    """
    count = 0
    for i in range(ood_step, len(scores)):
        if scores[i] >= threshold:
            count += 1
            if count >= patience:
                return i - ood_step
        else:
            count = 0
    return -1  # Không detect được
 
 
def compute_roc_curve(
    y_true: np.ndarray, y_scores: np.ndarray, n_thresholds: int = 200
) -> tuple[np.ndarray, np.ndarray]:
    """
    Tính điểm (FPR, TPR) cho ROC curve.
 
    Returns:
        (fpr_array, tpr_array)
    """
    thresholds = np.linspace(y_scores.min(), y_scores.max(), n_thresholds)
    fprs, tprs = [], []
    for t in thresholds:
        preds = (y_scores >= t).astype(int)
        tp = np.sum((preds == 1) & (y_true == 1))
        fp = np.sum((preds == 1) & (y_true == 0))
        fn = np.sum((preds == 0) & (y_true == 1))
        tn = np.sum((preds == 0) & (y_true == 0))
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        tprs.append(tpr)
        fprs.append(fpr)
    # Sắp xếp theo FPR tăng dần để vẽ đẹp
    order = np.argsort(fprs)
    return np.array(fprs)[order], np.array(tprs)[order]
 
 
# ═════════════════════════════════════════════════════════════════════════════
# ONLINE EMA NORMALIZER
# ═════════════════════════════════════════════════════════════════════════════
 
class OnlineEMANormalizer:
    """
    Chuẩn hóa surprise score theo thời gian thực (online) bằng EMA.
    Chỉ cập nhật trong giai đoạn ID (trước ood_step) — không bị "rò rỉ" OOD.
 
    Công thức:
        μ_t = α·μ_{t-1} + (1-α)·S_t
        σ²_t = α·σ²_{t-1} + (1-α)·(S_t - μ_t)²
        S̃_t = (S_t - μ_t) / (σ_t + ε)
    """
 
    def __init__(self, alpha: float = 0.99, eps: float = 1e-6):
        self.alpha = alpha
        self.eps = eps
        self.mu: float = 0.0
        self.var: float = 1.0
        self._initialized: bool = False
 
    def update_and_normalize(self, value: float, is_id: bool) -> float:
        """
        Cập nhật EMA nếu đang trong ID phase, rồi normalize value.
 
        Args:
            value:  surprise score raw
            is_id:  True nếu bước này là ID (dùng để cập nhật EMA)
 
        Returns:
            normalized score
        """
        if is_id:
            if not self._initialized:
                self.mu = value
                self.var = 1.0
                self._initialized = True
            else:
                self.mu = self.alpha * self.mu + (1 - self.alpha) * value
                self.var = self.alpha * self.var + (1 - self.alpha) * (value - self.mu) ** 2
 
        sigma = np.sqrt(self.var) + self.eps
        return (value - self.mu) / sigma
 
 
# ═════════════════════════════════════════════════════════════════════════════
# PREPROCESSING
# ═════════════════════════════════════════════════════════════════════════════
 
def preprocess_frame(obs: np.ndarray) -> np.ndarray:
    """RGB/Gray → Grayscale 84×84 uint8."""
    if obs.ndim == 3 and obs.shape[-1] == 3:
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    return cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
 
 
# ═════════════════════════════════════════════════════════════════════════════
# SINGLE SEED EVALUATION
# ═════════════════════════════════════════════════════════════════════════════
 
def run_single_seed(
    env_id: str,
    lewm: LeWorldModel,
    baseline: PixelPredictor,
    device: torch.device,
    ppo_model,
    steps: int,
    ood_step: int,
    new_frameskip: int,
    seed: int,
    ema_alpha: float = 0.99,
) -> dict:
    """
    Chạy một lượt evaluation trên một seed, trả về dict kết quả.
    """
    # Setup env
    if "procgen" in env_id.lower():
        env = gym.make(env_id, render_mode="rgb_array")
        env = ProcgenOODWrapper(env)
    else:
        env = gym.make(env_id, frameskip=1, render_mode="rgb_array")
        env = AtariDynamicFrameskipWrapper(env, initial_frameskip=4)
    env.action_space.seed(seed)
 
    # Buffers
    latent_surprises_raw: list[float] = []
    pixel_surprises_raw: list[float] = []
    latent_surprises_norm: list[float] = []
    pixel_surprises_norm: list[float] = []
    labels: list[int] = []
    rewards: list[float] = []
 
    lewm_normalizer = OnlineEMANormalizer(alpha=ema_alpha)
    pixel_normalizer = OnlineEMANormalizer(alpha=ema_alpha)
 
    # Frame stack cho PPO (4 frames × 84×84)
    frame_stack = np.zeros((1, 4, 84, 84), dtype=np.uint8)
 
    def push_frame(frame: np.ndarray):
        frame_stack[0, :-1] = frame_stack[0, 1:]
        frame_stack[0, -1] = frame
 
    obs, _ = env.reset(seed=seed)
    obs_proc = preprocess_frame(obs)
    push_frame(obs_proc)
 
    with torch.no_grad():
        for step in range(steps):
            is_id = step < ood_step
 
            # Chọn action
            if ppo_model is not None and HAS_PPO:
                action, _ = ppo_model.predict(frame_stack, deterministic=False)
                action_val = int(action[0])
            else:
                action_val = env.action_space.sample()
 
            # Chuẩn bị tensor obs_t
            obs_t = (
                torch.tensor(obs_proc, dtype=torch.float32)
                .unsqueeze(0).unsqueeze(0).to(device) / 255.0
            )
            act_t = torch.tensor([action_val], dtype=torch.long).to(device)
 
            # Kích hoạt OOD đúng tại ood_step
            if step == ood_step:
                if "procgen" in env_id.lower():
                    # Giả định procgen wrapper cần tham số momentum_prob
                    momentum_prob = 0.3 if new_frameskip > 0 else 0.0 # dùng new_frameskip như 1 cách trick
                    env.trigger_ood(momentum_prob=momentum_prob)
                else:
                    env.trigger_ood(new_frameskip=new_frameskip)
 
            # Step env
            next_obs, reward_val, terminated, truncated, _ = env.step(action_val)
            rewards.append(float(reward_val))
            done = terminated or truncated
 
            # Label
            labels.append(0 if is_id else 1)
 
            # Preprocess next_obs
            next_obs_proc = preprocess_frame(next_obs)
            next_obs_t = (
                torch.tensor(next_obs_proc, dtype=torch.float32)
                .unsqueeze(0).unsqueeze(0).to(device) / 255.0
            )
 
            push_frame(next_obs_proc)
 
            # ── Latent Surprise (LeWM) ──────────────────────────────────────
            z_t = lewm.get_latent(obs_t)
            z_t1 = lewm.get_latent(next_obs_t)
            pred_z_t1 = lewm.predict_next(z_t, act_t)
            l_surp_raw = float(torch.mean((pred_z_t1 - z_t1) ** 2).item())
            latent_surprises_raw.append(l_surp_raw)
            latent_surprises_norm.append(
                lewm_normalizer.update_and_normalize(l_surp_raw, is_id)
            )
 
            # ── Pixel Surprise (Baseline) ───────────────────────────────────
            pred_obs_t1 = baseline(obs_t, act_t)
            p_surp_raw = float(torch.mean((pred_obs_t1 - next_obs_t) ** 2).item())
            pixel_surprises_raw.append(p_surp_raw)
            pixel_surprises_norm.append(
                pixel_normalizer.update_and_normalize(p_surp_raw, is_id)
            )
 
            obs_proc = next_obs_proc
 
            if done:
                cur_fs = env.frameskip
                obs, _ = env.reset()
                env.frameskip = cur_fs
                obs_proc = preprocess_frame(obs)
                push_frame(obs_proc)
 
    env.close()
 
    y_true = np.array(labels)
    l_norm = np.array(latent_surprises_norm)
    p_norm = np.array(pixel_surprises_norm)
 
    # ── Metrics ──────────────────────────────────────────────────────────────
    lewm_auroc = compute_auroc(y_true, l_norm)
    lewm_f1, lewm_thresh = compute_max_f1(y_true, l_norm)
    lewm_ece = compute_ece(y_true, l_norm)
    lewm_delay = compute_detection_delay(y_true, l_norm, lewm_thresh, ood_step)
 
    base_auroc = compute_auroc(y_true, p_norm)
    base_f1, base_thresh = compute_max_f1(y_true, p_norm)
    base_ece = compute_ece(y_true, p_norm)
    base_delay = compute_detection_delay(y_true, p_norm, base_thresh, ood_step)
 
    lewm_fpr, lewm_tpr = compute_roc_curve(y_true, l_norm)
    base_fpr, base_tpr = compute_roc_curve(y_true, p_norm)
 
    # Smoothing reward
    rewards_arr = np.array(rewards)
    smoothed_rewards = np.zeros_like(rewards_arr)
    if len(rewards_arr) > 0:
        smoothed_rewards[0] = rewards_arr[0]
        alpha = 0.1
        for i in range(1, len(rewards_arr)):
            smoothed_rewards[i] = alpha * rewards_arr[i] + (1 - alpha) * smoothed_rewards[i - 1]

    # Calculate Pearson correlation between surprise and reward in OOD phase
    ood_mask = y_true == 1
    if ood_mask.sum() > 2:
        try:
            import scipy.stats as stats
            lewm_corr, _ = stats.pearsonr(l_norm[ood_mask], smoothed_rewards[ood_mask])
            base_corr, _ = stats.pearsonr(p_norm[ood_mask], smoothed_rewards[ood_mask])
        except Exception:
            lewm_corr, base_corr = 0.0, 0.0
    else:
        lewm_corr, base_corr = 0.0, 0.0
 
    return {
        "rewards": rewards_arr,
        "smoothed_rewards": smoothed_rewards,
        "lewm_corr": lewm_corr,
        "base_corr": base_corr,
        # Raw arrays (để plot)
        "l_norm": l_norm,
        "p_norm": p_norm,
        "labels": y_true,
        "lewm_fpr": lewm_fpr,
        "lewm_tpr": lewm_tpr,
        "base_fpr": base_fpr,
        "base_tpr": base_tpr,
        # Scalar metrics
        "lewm_auroc": lewm_auroc,
        "lewm_f1": lewm_f1,
        "lewm_ece": lewm_ece,
        "lewm_delay": lewm_delay,
        "base_auroc": base_auroc,
        "base_f1": base_f1,
        "base_ece": base_ece,
        "base_delay": base_delay,
    }
 
 
# ═════════════════════════════════════════════════════════════════════════════
# MULTI-SEED RUNNER
# ═════════════════════════════════════════════════════════════════════════════
 
def run_multi_seed(
    env_id: str,
    lewm: LeWorldModel,
    baseline: PixelPredictor,
    device: torch.device,
    ppo_model,
    steps: int,
    ood_step: int,
    new_frameskip: int,
    n_seeds: int,
    ema_alpha: float,
) -> dict:
    """
    Chạy evaluation trên n_seeds seed khác nhau, trả về mean ± std.
    """
    all_results = []
    for seed in range(n_seeds):
        print(f"  [Seed {seed+1}/{n_seeds}] frameskip={new_frameskip} ...", flush=True)
        r = run_single_seed(
            env_id, lewm, baseline, device, ppo_model,
            steps, ood_step, new_frameskip, seed, ema_alpha
        )
        all_results.append(r)
 
    def agg(key):
        vals = [r[key] for r in all_results]
        return float(np.mean(vals)), float(np.std(vals))
 
    return {
        "n_seeds": n_seeds,
        "frameskip": new_frameskip,
        # LeWM
        "lewm_auroc_mean": agg("lewm_auroc")[0],
        "lewm_auroc_std":  agg("lewm_auroc")[1],
        "lewm_f1_mean":    agg("lewm_f1")[0],
        "lewm_f1_std":     agg("lewm_f1")[1],
        "lewm_ece_mean":   agg("lewm_ece")[0],
        "lewm_ece_std":    agg("lewm_ece")[1],
        "lewm_delay_mean": agg("lewm_delay")[0],
        "lewm_delay_std":  agg("lewm_delay")[1],
        "lewm_corr_mean":  agg("lewm_corr")[0],
        "lewm_corr_std":   agg("lewm_corr")[1],
        # Baseline
        "base_auroc_mean": agg("base_auroc")[0],
        "base_auroc_std":  agg("base_auroc")[1],
        "base_f1_mean":    agg("base_f1")[0],
        "base_f1_std":     agg("base_f1")[1],
        "base_ece_mean":   agg("base_ece")[0],
        "base_ece_std":    agg("base_ece")[1],
        "base_delay_mean": agg("base_delay")[0],
        "base_delay_std":  agg("base_delay")[1],
        "base_corr_mean":  agg("base_corr")[0],
        "base_corr_std":   agg("base_corr")[1],
        # Raw arrays từ seed cuối (để vẽ representative plot)
        "_last_result": all_results[-1],
    }
 
 
# ═════════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═════════════════════════════════════════════════════════════════════════════
 
def plot_results(
    results_by_fs: dict,
    ood_step: int,
    env_id: str,
    save_path: str,
):
    """
    Vẽ 2 loại biểu đồ:
      (A) Surprise trajectory: LeWM vs Baseline qua thời gian
      (B) ROC curve: LeWM vs Baseline cho từng frameskip variant
    """
    n_variants = len(results_by_fs)
    fig, axes = plt.subplots(
        nrows=n_variants,
        ncols=3,
        figsize=(20, 4 * n_variants),
    )
    if n_variants == 1:
        axes = np.array([axes])  # Đảm bảo 2D indexing
 
    for row_idx, (fs_val, agg) in enumerate(results_by_fs.items()):
        last = agg["_last_result"]
        steps_total = len(last["l_norm"])
        x = np.arange(steps_total)
 
        # ── Panel A: Surprise Trajectory ─────────────────────────────────
        ax_traj = axes[row_idx, 0]
        ax_traj.plot(x, last["l_norm"],  color="#1f77b4", lw=1.5,
                     label=f"LeWM (AUROC={agg['lewm_auroc_mean']:.3f})")
        ax_traj.plot(x, last["p_norm"],  color="#ff7f0e", lw=1.5, alpha=0.7,
                     label=f"Baseline (AUROC={agg['base_auroc_mean']:.3f})")
        ax_traj.axvline(x=ood_step, color="red", ls="--", lw=2,
                        label=f"OOD trigger (fs→{fs_val})")
        ax_traj.set_title(f"Surprise Trajectory | frameskip→{fs_val}", fontsize=12, fontweight="bold")
        ax_traj.set_xlabel("Step")
        ax_traj.set_ylabel("Normalized Surprise (EMA)")
        ax_traj.legend(loc="upper left", fontsize=9)
        ax_traj.grid(True, ls=":", alpha=0.5)
 
        # ── Panel B: ROC Curve ────────────────────────────────────────────
        ax_roc = axes[row_idx, 1]
        ax_roc.plot(last["lewm_fpr"], last["lewm_tpr"], color="#1f77b4", lw=2,
                    label=f"LeWM AUROC={agg['lewm_auroc_mean']:.3f}±{agg['lewm_auroc_std']:.3f}")
        ax_roc.plot(last["base_fpr"], last["base_tpr"], color="#ff7f0e", lw=2,
                    label=f"Baseline AUROC={agg['base_auroc_mean']:.3f}±{agg['base_auroc_std']:.3f}")
        ax_roc.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random")
        ax_roc.set_title(f"ROC Curve | frameskip→{fs_val}", fontsize=12, fontweight="bold")
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.legend(loc="lower right", fontsize=9)
        ax_roc.grid(True, ls=":", alpha=0.5)
        ax_roc.set_xlim([-0.02, 1.02])
        ax_roc.set_ylim([-0.02, 1.02])
 
        # ── Panel C: Return Degradation ──────────────────────────────────
        ax_rew = axes[row_idx, 2]
        ax_rew.plot(x, last["smoothed_rewards"], color="green", lw=1.5,
                    label=f"Smoothed Reward\nCorr(LeWM): {agg['lewm_corr_mean']:.2f}")
        ax_rew.axvline(x=ood_step, color="red", ls="--", lw=2,
                       label=f"OOD trigger")
        ax_rew.set_title(f"Return Degradation | frameskip/OOD→{fs_val}", fontsize=12, fontweight="bold")
        ax_rew.set_xlabel("Step")
        ax_rew.set_ylabel("Smoothed Reward")
        ax_rew.legend(loc="best", fontsize=9)
        ax_rew.grid(True, ls=":", alpha=0.5)

    plt.suptitle(
        f"OOD Dynamics Detection — {env_id}\n"
        f"LeWM (JEPA) vs Pixel Reconstruction Baseline",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Đã lưu tại: {save_path}")
 
 
# ═════════════════════════════════════════════════════════════════════════════
# REPORT PRINTER
# ═════════════════════════════════════════════════════════════════════════════
 
SEPARATOR = "=" * 84
 
def print_and_save_report(
    results_by_fs: dict,
    env_id: str,
    steps: int,
    ood_step: int,
    n_seeds: int,
    save_path: str,
):
    """In kết quả ra console và lưu file .txt theo định dạng chuẩn paper."""
    lines = []
    lines.append(SEPARATOR)
    lines.append("OOD DYNAMICS DETECTION — EVALUATION REPORT")
    lines.append(SEPARATOR)
    lines.append(f"Environment : {env_id}")
    lines.append(f"Total Steps : {steps}  |  OOD Trigger : step {ood_step}")
    lines.append(f"Seeds       : {n_seeds}  (mean ± std)")
    lines.append("")
 
    header = f"{'Variant':<12} {'Model':<10} {'AUROC':>10} {'F1':>10} {'ECE':>10} {'Delay':>10} {'Corr':>10}"
    lines.append(header)
    lines.append("-" * 84)
 
    for fs_val, agg in results_by_fs.items():
        variant = f"OOD→{fs_val}"
        lewm_row = (
            f"{variant:<12} {'LeWM':<10} "
            f"{agg['lewm_auroc_mean']:.4f}±{agg['lewm_auroc_std']:.4f}  "
            f"{agg['lewm_f1_mean']:.4f}±{agg['lewm_f1_std']:.4f}  "
            f"{agg['lewm_ece_mean']:.4f}±{agg['lewm_ece_std']:.4f}  "
            f"{agg['lewm_delay_mean']:.1f}±{agg['lewm_delay_std']:.1f}  "
            f"{agg['lewm_corr_mean']:.3f}±{agg['lewm_corr_std']:.3f}"
        )
        base_row = (
            f"{'':<12} {'Baseline':<10} "
            f"{agg['base_auroc_mean']:.4f}±{agg['base_auroc_std']:.4f}  "
            f"{agg['base_f1_mean']:.4f}±{agg['base_f1_std']:.4f}  "
            f"{agg['base_ece_mean']:.4f}±{agg['base_ece_std']:.4f}  "
            f"{agg['base_delay_mean']:.1f}±{agg['base_delay_std']:.1f}  "
            f"{agg['base_corr_mean']:.3f}±{agg['base_corr_std']:.3f}"
        )
        lines.append(lewm_row)
        lines.append(base_row)
        lines.append("-" * 84)
 
    lines.append("")
    lines.append("Metrics:")
    lines.append("  AUROC  : Area Under ROC Curve. Higher = better. Random = 0.5")
    lines.append("  F1     : Best F1 over all thresholds. Higher = better.")
    lines.append("  ECE    : Expected Calibration Error. Lower = better.")
    lines.append("  Delay  : Steps after OOD trigger before detection. Lower = better.")
    lines.append("           -1 means detection failed.")
    lines.append("  Corr   : Pearson correlation between Latent Surprise and Smoothed Reward")
    lines.append("           during OOD phase. Stronger negative correlation = better.")
    lines.append("=" * 84)
 
    report_str = "\n".join(lines)
    print(report_str)
 
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(report_str)
    print(f"[Report] Đã lưu tại: {save_path}")
 
 
# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
 
def main(args):
    clean_id = args.env.replace("/", "_").replace("-", "_")
    results_dir = os.path.join(project_dir, "results")
    models_dir = os.path.join(project_dir, "models")
    os.makedirs(results_dir, exist_ok=True)
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
 
    # ── Xác định action_dim ──────────────────────────────────────────────────
    action_dim = 4 if "Breakout" in args.env else 6
 
    # ── Load models ──────────────────────────────────────────────────────────
    lewm_path = os.path.join(models_dir, f"lewm_{clean_id}.pth")
    base_path = os.path.join(models_dir, f"baseline_{clean_id}.pth")
 
    if not os.path.exists(lewm_path):
        raise FileNotFoundError(
            f"Không tìm thấy LeWM checkpoint tại: {lewm_path}\n"
            "Chạy train_lewm.py trước!"
        )
    if not os.path.exists(base_path):
        raise FileNotFoundError(
            f"Không tìm thấy Baseline checkpoint tại: {base_path}\n"
            "Chạy train_lewm.py trước!"
        )
 
    print(f"Loading LeWM từ: {lewm_path}")
    lewm = LeWorldModel(latent_dim=args.latent_dim, action_dim=action_dim).to(device)
    lewm.load_state_dict(torch.load(lewm_path, map_location=device, weights_only=True))
    lewm.eval()
 
    print(f"Loading Baseline từ: {base_path}")
    baseline = PixelPredictor(action_dim=action_dim).to(device)
    baseline.load_state_dict(torch.load(base_path, map_location=device, weights_only=True))
    baseline.eval()
 
    # ── Load PPO (optional) ──────────────────────────────────────────────────
    ppo_model = None
    ppo_path = os.path.join(models_dir, f"ppo_{clean_id}.zip")
    if HAS_PPO and os.path.exists(ppo_path):
        print(f"Loading PPO Agent từ: {ppo_path}")
        ppo_model = PPO.load(ppo_path)
    else:
        print("Cảnh báo: Dùng Random Agent (PPO không tìm thấy).")
 
    # ── Ablation over frameskip variants ────────────────────────────────────
    frameskip_variants = args.frameskip_variants
    print(f"\nAblation variants: frameskip → {frameskip_variants}")
    print(f"Seeds: {args.seeds} | Steps: {args.steps} | OOD at step: {args.ood_step}\n")
 
    results_by_fs = {}
    for fs_val in frameskip_variants:
        print(f"\n{'─'*60}")
        print(f"Variant: frameskip → {fs_val}")
        print(f"{'─'*60}")
        agg = run_multi_seed(
            env_id=args.env,
            lewm=lewm,
            baseline=baseline,
            device=device,
            ppo_model=ppo_model,
            steps=args.steps,
            ood_step=args.ood_step,
            new_frameskip=fs_val,
            n_seeds=args.seeds,
            ema_alpha=args.ema_alpha,
        )
        results_by_fs[fs_val] = agg
 
    # ── Save JSON ────────────────────────────────────────────────────────────
    json_path = os.path.join(results_dir, f"ood_metrics_{clean_id}.json")
    # Loại bỏ numpy arrays trước khi serialize JSON
    json_safe = {}
    for fs_val, agg in results_by_fs.items():
        d = {k: v for k, v in agg.items() if not k.startswith("_")}
        json_safe[str(fs_val)] = d
 
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_safe, f, indent=2)
    print(f"\n[JSON] Metrics đã lưu tại: {json_path}")
 
    # ── Plot ─────────────────────────────────────────────────────────────────
    plot_path = os.path.join(results_dir, f"ood_plot_{clean_id}.png")
    plot_results(results_by_fs, args.ood_step, args.env, plot_path)
 
    # ── Text report ──────────────────────────────────────────────────────────
    txt_path = os.path.join(results_dir, f"ood_summary_{clean_id}.txt")
    print_and_save_report(
        results_by_fs, args.env, args.steps, args.ood_step, args.seeds, txt_path
    )
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Full OOD Dynamics Evaluation Pipeline — LeWM vs Baseline"
    )
    parser.add_argument("--env", type=str, default="ALE/Pong-v5",
                        help="Atari Environment ID (default: ALE/Pong-v5)")
    parser.add_argument("--steps", type=int, default=600,
                        help="Tổng số steps mỗi episode (default: 600)")
    parser.add_argument("--ood_step", type=int, default=300,
                        help="Bước kích hoạt OOD dynamics (default: 300)")
    parser.add_argument("--frameskip_variants", type=int, nargs="+", default=[2, 8],
                        help="Danh sách frameskip OOD để ablation (default: 2 8)")
    parser.add_argument("--seeds", type=int, default=3,
                        help="Số lượng seed để tính mean±std (default: 3)")
    parser.add_argument("--latent_dim", type=int, default=64,
                        help="Chiều latent của LeWM (default: 64)")
    parser.add_argument("--ema_alpha", type=float, default=0.99,
                        help="Hệ số EMA normalization (default: 0.99)")
 
    args = parser.parse_args()
 
    # Validate
    assert args.ood_step < args.steps, \
        f"ood_step ({args.ood_step}) phải nhỏ hơn steps ({args.steps})"
    assert args.seeds >= 1, "seeds phải >= 1"
 
    main(args)
