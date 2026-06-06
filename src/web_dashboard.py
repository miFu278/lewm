import torch
import numpy as np
import os
import cv2
import json
import time
import threading
import argparse
import sys
from flask import Flask, Response, render_template_string, request, jsonify
from stable_baselines3 import PPO
import ale_py
import gymnasium as gym

# Thêm đường dẫn project vào sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
if project_dir not in sys.path:
    sys.path.append(project_dir)

from src.env.atari_ood_wrapper import AtariDynamicFrameskipWrapper
from src.models.lewm import LeWorldModel
from src.models.baseline import PixelPredictor

# Flask App
app = Flask(__name__)

# Trạng thái toàn cục (Global State) chia sẻ giữa Flask và Game Loop Thread
state_lock = threading.Lock()
current_frame = None
current_metrics = {
    "step": 0,
    "action": 0,
    "latent_surprise": 0.0,
    "pixel_surprise": 0.0,
    "norm_latent": 0.0,
    "norm_pixel": 0.0,
    "is_ood": False,
    "frameskip": 4
}
command_queue = []

# Cấu hình Web Page (Glassmorphism & Real-time Charts)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LeWorldModel (JEPA) - OOD Dynamics Detector</title>
    <!-- Google Fonts -->
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap" rel="stylesheet">
    <!-- Chart.js CDN -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {
            --bg-gradient: radial-gradient(circle at top, #0f172a 0%, #020617 100%);
            --glass-bg: rgba(15, 23, 42, 0.45);
            --glass-border: rgba(255, 255, 255, 0.08);
            --neon-blue: #06b6d4;
            --neon-blue-glow: rgba(6, 182, 212, 0.4);
            --neon-orange: #f97316;
            --neon-orange-glow: rgba(249, 115, 22, 0.4);
            --neon-green: #10b981;
            --neon-green-glow: rgba(16, 185, 129, 0.5);
            --neon-red: #ef4444;
            --neon-red-glow: rgba(239, 68, 68, 0.5);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Outfit', sans-serif;
            background: var(--bg-gradient);
            color: #f8fafc;
            min-height: 100vh;
            overflow-x: hidden;
            padding: 2rem;
        }

        header {
            text-align: center;
            margin-bottom: 2rem;
        }

        header h1 {
            font-size: 2.2rem;
            font-weight: 700;
            background: linear-gradient(to right, #22d3ee, #818cf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }

        header p {
            font-size: 1rem;
            color: #94a3b8;
            font-weight: 300;
        }

        .dashboard-container {
            max-width: 1400px;
            margin: 0 auto;
            display: grid;
            grid-template-columns: 1fr 1.2fr;
            gap: 2rem;
        }

        @media (max-width: 1024px) {
            .dashboard-container {
                grid-template-columns: 1fr;
            }
        }

        .panel {
            background: var(--glass-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--glass-border);
            border-radius: 20px;
            padding: 1.5rem;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        }

        .panel-title {
            font-size: 1.2rem;
            font-weight: 600;
            margin-bottom: 1.2rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            padding-bottom: 0.5rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        /* Thẻ trạng thái OOD */
        .status-badge {
            padding: 0.4rem 1rem;
            border-radius: 30px;
            font-size: 0.85rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            transition: all 0.3s ease;
        }

        .status-id {
            background: rgba(16, 185, 129, 0.1);
            border: 1px solid var(--neon-green);
            color: var(--neon-green);
            box-shadow: 0 0 10px var(--neon-green-glow);
        }

        .status-ood {
            background: rgba(239, 68, 68, 0.1);
            border: 1px solid var(--neon-red);
            color: var(--neon-red);
            box-shadow: 0 0 10px var(--neon-red-glow);
            animation: pulse 1.5s infinite;
        }

        @keyframes pulse {
            0% { box-shadow: 0 0 5px var(--neon-red-glow); }
            50% { box-shadow: 0 0 20px var(--neon-red-glow); }
            100% { box-shadow: 0 0 5px var(--neon-red-glow); }
        }

        /* Màn hình game stream */
        .screen-container {
            display: flex;
            justify-content: center;
            align-items: center;
            background: #000;
            border-radius: 12px;
            overflow: hidden;
            border: 2px solid rgba(255, 255, 255, 0.05);
            aspect-ratio: 4/3;
            margin-bottom: 1.5rem;
        }

        .screen-container img {
            width: 100%;
            height: 100%;
            object-fit: contain;
        }

        /* Bộ điều khiển */
        .controls-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1rem;
            margin-bottom: 1.5rem;
        }

        .btn {
            font-family: 'Outfit', sans-serif;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: #e2e8f0;
            padding: 0.8rem 1rem;
            border-radius: 10px;
            cursor: pointer;
            font-weight: 600;
            font-size: 0.9rem;
            transition: all 0.2s ease;
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 0.5rem;
        }

        .btn:hover {
            background: rgba(255, 255, 255, 0.12);
            border-color: rgba(255, 255, 255, 0.2);
            transform: translateY(-2px);
        }

        .btn-active {
            background: linear-gradient(135deg, #4f46e5 0%, #3b82f6 100%);
            border-color: #60a5fa;
            box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4);
        }

        .btn-danger {
            background: rgba(239, 68, 68, 0.15);
            border-color: rgba(239, 68, 68, 0.4);
            color: #fca5a5;
        }

        .btn-danger:hover {
            background: rgba(239, 68, 68, 0.25);
            border-color: var(--neon-red);
            box-shadow: 0 4px 15px rgba(239, 68, 68, 0.3);
        }

        /* Bảng hiển thị thông số số liệu */
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1rem;
        }

        .metric-card {
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            padding: 1rem;
            text-align: center;
        }

        .metric-label {
            font-size: 0.8rem;
            color: #64748b;
            text-transform: uppercase;
            margin-bottom: 0.3rem;
            letter-spacing: 0.05em;
        }

        .metric-value {
            font-size: 1.5rem;
            font-weight: 700;
        }

        .metric-lewm {
            color: var(--neon-blue);
            text-shadow: 0 0 10px var(--neon-blue-glow);
        }

        .metric-base {
            color: var(--neon-orange);
            text-shadow: 0 0 10px var(--neon-orange-glow);
        }

        /* Biểu đồ */
        .chart-panel {
            display: flex;
            flex-direction: column;
            height: 100%;
        }

        .chart-container {
            flex-grow: 1;
            position: relative;
            min-height: 400px;
            width: 100%;
        }
    </style>
</head>
<body>

    <header>
        <h1>LeWorldModel (JEPA) Interactive Dashboard</h1>
        <p>Giám sát trực quan trạng thái Out-of-Distribution Dynamics của môi trường Atari trong thời gian thực</p>
    </header>

    <div class="dashboard-container">
        <!-- Panel bên trái: Game Screen & Controls -->
        <div class="panel">
            <div class="panel-title">
                <span>Trực Tiếp Giả Lập Game</span>
                <span id="status-badge" class="status-badge status-id">In-Distribution</span>
            </div>

            <!-- Màn hình game -->
            <div class="screen-container">
                <img src="/video_feed" alt="Atari Live Stream">
            </div>

            <!-- Panel điều khiển OOD -->
            <div class="panel-title">Can Thiệp Dynamics (OOD Control)</div>
            <div class="controls-grid">
                <button id="btn-fs4" class="btn btn-active" onclick="setFrameskip(4)">Chuẩn (Frameskip = 4)</button>
                <button id="btn-fs2" class="btn" onclick="setFrameskip(2)">Nhanh x2 (Frameskip = 2)</button>
                <button id="btn-fs1" class="btn" onclick="setFrameskip(1)">Nhanh x4 (Frameskip = 1)</button>
                <button id="btn-fs8" class="btn" onclick="setFrameskip(8)">Chậm x2 (Frameskip = 8)</button>
            </div>
            <div style="margin-bottom: 1.5rem;">
                <button class="btn btn-danger" style="width: 100%;" onclick="resetEnv()">Reset Môi Trường (Restart Game)</button>
            </div>

            <!-- Thông số tức thời -->
            <div class="panel-title">Thông Số Tức Thời</div>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">Mẫu Đang Chạy (Step)</div>
                    <div id="metric-step" class="metric-value">0</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Frameskip Hiện Tại</div>
                    <div id="metric-fs" class="metric-value">4</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">LeWM Latent Surprise</div>
                    <div id="metric-lewm" class="metric-value metric-lewm">0.00</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Baseline Pixel Surprise</div>
                    <div id="metric-base" class="metric-value metric-base">0.00</div>
                </div>
            </div>
        </div>

        <!-- Panel bên phải: Real-time Chart -->
        <div class="panel chart-panel">
            <div class="panel-title">Đồ Thị Surprise Scores Chuẩn Hóa (Real-Time)</div>
            <div class="chart-container">
                <canvas id="surpriseChart"></canvas>
            </div>
        </div>
    </div>

    <script>
        // Cấu hình Chart.js
        const ctx = document.getElementById('surpriseChart').getContext('2d');
        const maxDataPoints = 100;
        
        const chartData = {
            labels: [],
            datasets: [
                {
                    label: 'LeWorldModel Latent Surprise (JEPA)',
                    borderColor: '#06b6d4',
                    backgroundColor: 'rgba(6, 182, 212, 0.05)',
                    borderWidth: 3,
                    pointRadius: 0,
                    data: [],
                    tension: 0.2,
                    fill: true
                },
                {
                    label: 'Pixel Predictor Surprise (Baseline)',
                    borderColor: '#f97316',
                    backgroundColor: 'rgba(249, 115, 22, 0.05)',
                    borderWidth: 2,
                    pointRadius: 0,
                    data: [],
                    tension: 0.2,
                    fill: true
                }
            ]
        };

        const chartOptions = {
            responsive: true,
            maintainAspectRatio: false,
            animation: {
                duration: 0 // Tắt animation để vẽ mượt hơn khi stream nhanh
            },
            scales: {
                x: {
                    grid: {
                        color: 'rgba(255, 255, 255, 0.05)'
                    },
                    ticks: {
                        color: '#64748b'
                    },
                    title: {
                        display: true,
                        text: 'Time Step',
                        color: '#64748b'
                    }
                },
                y: {
                    grid: {
                        color: 'rgba(255, 255, 255, 0.05)'
                    },
                    ticks: {
                        color: '#64748b'
                    },
                    title: {
                        display: true,
                        text: 'Normalized Surprise',
                        color: '#64748b'
                    }
                }
            },
            plugins: {
                legend: {
                    position: 'top',
                    labels: {
                        color: '#f8fafc',
                        font: {
                            family: 'Outfit'
                        }
                    }
                }
            }
        };

        const surpriseChart = new Chart(ctx, {
            type: 'line',
            data: chartData,
            options: chartOptions
        });

        // Kết nối Server-Sent Events (SSE) để stream metrics liên tục
        const eventSource = new EventSource('/metrics');
        
        eventSource.onmessage = function(event) {
            const metrics = JSON.parse(event.data);
            
            // Cập nhật giao diện số liệu
            document.getElementById('metric-step').innerText = metrics.step;
            document.getElementById('metric-fs').innerText = metrics.frameskip;
            document.getElementById('metric-lewm').innerText = metrics.norm_latent.toFixed(2);
            document.getElementById('metric-base').innerText = metrics.norm_pixel.toFixed(2);
            
            // Cập nhật trạng thái nhãn OOD
            const badge = document.getElementById('status-badge');
            if (metrics.is_ood) {
                badge.innerText = 'OOD (Anomaly Dynamics)';
                badge.className = 'status-badge status-ood';
            } else {
                badge.innerText = 'In-Distribution';
                badge.className = 'status-badge status-id';
            }
            
            // Cập nhật đồ thị Chart.js
            surpriseChart.data.labels.push(metrics.step);
            surpriseChart.data.datasets[0].data.push(metrics.norm_latent);
            surpriseChart.data.datasets[1].data.push(metrics.norm_pixel);
            
            // Giới hạn số lượng điểm hiển thị
            if (surpriseChart.data.labels.length > maxDataPoints) {
                surpriseChart.data.labels.shift();
                surpriseChart.data.datasets[0].data.shift();
                surpriseChart.data.datasets[1].data.shift();
            }
            
            surpriseChart.update();
        };

        // Hàm gọi API đặt Frameskip
        function setFrameskip(fs) {
            fetch('/trigger_action', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ action: 'set_frameskip', value: fs })
            })
            .then(res => res.json())
            .then(data => {
                // Đổi trạng thái active cho nút bấm
                const buttons = ['btn-fs1', 'btn-fs2', 'btn-fs4', 'btn-fs8'];
                buttons.forEach(id => {
                    document.getElementById(id).classList.remove('btn-active');
                });
                document.getElementById('btn-fs' + fs).classList.add('btn-active');
            });
        }

        // Hàm gọi API reset environment
        function resetEnv() {
            fetch('/trigger_action', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ action: 'reset' })
            })
            .then(res => res.json())
            .then(data => {
                // Xóa sạch dữ liệu trên đồ thị để vẽ lại từ đầu
                surpriseChart.data.labels = [];
                surpriseChart.data.datasets[0].data = [];
                surpriseChart.data.datasets[1].data = [];
                surpriseChart.update();
                
                // Đặt lại nút active về frameskip = 4
                const buttons = ['btn-fs1', 'btn-fs2', 'btn-fs4', 'btn-fs8'];
                buttons.forEach(id => {
                    document.getElementById(id).classList.remove('btn-active');
                });
                document.getElementById('btn-fs4').classList.add('btn-active');
            });
        }
    </script>
</body>
</html>
"""

# Vòng lặp giả lập và inference chạy ngầm (Background Simulation Thread)
def run_simulation(env_id, latent_dim):
    global current_frame, current_metrics, command_queue
    
    clean_env_id = env_id.replace("/", "_").replace("-", "_")
    models_dir = os.path.join(project_dir, "models")
    
    # 1. Đường dẫn models
    ppo_path = os.path.join(models_dir, f"ppo_{clean_env_id}.zip")
    lewm_path = os.path.join(models_dir, f"lewm_{clean_env_id}.pth")
    baseline_path = os.path.join(models_dir, f"baseline_{clean_env_id}.pth")
    
    # 2. Khởi tạo môi trường
    print(f"[Engine] Khởi chạy môi trường {env_id}...")
    raw_env = gym.make(env_id, frameskip=1, render_mode="rgb_array")
    env = AtariDynamicFrameskipWrapper(raw_env, initial_frameskip=4)
    
    # Xác định số lượng hành động
    action_dim = 6
    if "Breakout" in env_id:
        action_dim = 4
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Tải PPO Agent để chọn hành động thông minh
    ppo_model = None
    if os.path.exists(ppo_path):
        print(f"[Engine] Đang tải PPO Agent từ {ppo_path}...")
        ppo_model = PPO.load(ppo_path)
    else:
        print("[Engine] Không tìm thấy PPO Agent. Sẽ sử dụng hành động ngẫu nhiên!")
        
    # Tải LeWM và Baseline
    if not os.path.exists(lewm_path) or not os.path.exists(baseline_path):
        print("[Engine] Lỗi nghiêm trọng: Chưa có file model weight cho LeWM hoặc Baseline!")
        print("[Engine] Vui lòng chạy train_lewm.py trước!")
        return
        
    print("[Engine] Đang tải mô hình LeWorldModel...")
    lewm = LeWorldModel(latent_dim=latent_dim, action_dim=action_dim).to(device)
    lewm.load_state_dict(torch.load(lewm_path, map_location=device))
    lewm.eval()
    
    print("[Engine] Đang tải mô hình Pixel Predictor Baseline...")
    baseline = PixelPredictor(action_dim=action_dim).to(device)
    baseline.load_state_dict(torch.load(baseline_path, map_location=device))
    baseline.eval()
    
    # Thống kê chạy ngầm cho Normalization (EMA)
    # Ta dùng một bộ đệm ban đầu để tính Mean/Std chuẩn hóa
    latent_history = []
    pixel_history = []
    
    running_mean_latent = 0.0
    running_std_latent = 1.0
    running_mean_pixel = 0.0
    running_std_pixel = 1.0
    
    # Hàm reset hoàn toàn môi trường và tính toán chuẩn hóa
    def init_simulation():
        nonlocal latent_history, pixel_history, running_mean_latent, running_std_latent, running_mean_pixel, running_std_pixel
        obs, info = env.reset()
        latent_history = []
        pixel_history = []
        running_mean_latent = 0.0
        running_std_latent = 1.0
        running_mean_pixel = 0.0
        running_std_pixel = 1.0
        return obs
        
    obs = init_simulation()
    
    # Bộ đệm Frame Stack (4 frames) cho PPO
    frame_stack = np.zeros((1, 4, 84, 84), dtype=np.uint8)
    
    def update_frame_stack(frame):
        frame_stack[0, :-1] = frame_stack[0, 1:]
        frame_stack[0, -1] = frame
        
    def preprocess_frame(obs_img):
        if len(obs_img.shape) == 3 and obs_img.shape[-1] == 3:
            obs_gray = cv2.cvtColor(obs_img, cv2.COLOR_RGB2GRAY)
        else:
            obs_gray = obs_img
        obs_resized = cv2.resize(obs_gray, (84, 84), interpolation=cv2.INTER_AREA)
        return obs_resized

    obs_processed = preprocess_frame(obs)
    update_frame_stack(obs_processed)
    
    step_count = 0
    
    # Vòng lặp chính vô hạn
    while True:
        # Xử lý các câu lệnh nhận về từ Web Dashboard (Reset hoặc Đổi frameskip)
        if len(command_queue) > 0:
            cmd = command_queue.pop(0)
            if cmd["type"] == "reset":
                print("[Engine] Nhận yêu cầu restart game...")
                obs = init_simulation()
                obs_processed = preprocess_frame(obs)
                update_frame_stack(obs_processed)
                step_count = 0
                env.frameskip = 4 # Reset về mặc định
                continue
            elif cmd["type"] == "set_frameskip":
                new_fs = cmd["value"]
                env.trigger_ood(new_fs)
                
        # Chọn hành động
        if ppo_model is not None:
            action, _ = ppo_model.predict(frame_stack, deterministic=False)
            action_val = int(action[0])
        else:
            action_val = env.action_space.sample()
            
        # Tiền xử lý state hiện tại
        obs_t_tensor = torch.tensor(obs_processed, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device) / 255.0
        act_t_tensor = torch.tensor([action_val], dtype=torch.long).to(device)
        
        # Bước đi trong game
        # Lấy hình ảnh thô để stream lên giao diện (env.render trả về RGB)
        frame_rgb = env.render()
        
        next_obs, reward, terminated, truncated, info = env.step(action_val)
        done = terminated or truncated
        
        next_obs_processed = preprocess_frame(next_obs)
        next_obs_tensor = torch.tensor(next_obs_processed, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device) / 255.0
        
        # Cập nhật Frame Stack cho PPO
        update_frame_stack(next_obs_processed)
        
        # --- Inference các mô hình & Tính toán metrics ---
        with torch.no_grad():
            # 1. LeWM Latent Surprise
            z_t = lewm.get_latent(obs_t_tensor)
            z_t1 = lewm.get_latent(next_obs_tensor)
            pred_z_t1 = lewm.predict_next(z_t, act_t_tensor)
            latent_surprise = torch.mean((pred_z_t1 - z_t1) ** 2).item()
            
            # 2. Pixel Predictor Surprise
            pred_obs_t1 = baseline(obs_t_tensor, act_t_tensor)
            pixel_surprise = torch.mean((pred_obs_t1 - next_obs_tensor) ** 2).item()
            
        # Chuẩn hóa (Normalization)
        # Để tránh việc chuẩn hóa bị loãng khi gặp OOD, ta chỉ cập nhật thống kê Mean/Std khi hệ thống ở trạng thái In-Distribution (Mặc định frameskip = 4)
        is_currently_ood = (env.frameskip != 4)
        
        if not is_currently_ood:
            # Lưu trữ history tối đa 150 bước ID để làm mốc thống kê chuẩn hóa
            latent_history.append(latent_surprise)
            pixel_history.append(pixel_surprise)
            if len(latent_history) > 150:
                latent_history.pop(0)
                pixel_history.pop(0)
                
            running_mean_latent = np.mean(latent_history)
            running_std_latent = np.std(latent_history) + 1e-6
            running_mean_pixel = np.mean(pixel_history)
            running_std_pixel = np.std(pixel_history) + 1e-6
            
        # Tính toán Surprise đã chuẩn hóa
        norm_latent = (latent_surprise - running_mean_latent) / running_std_latent
        norm_pixel = (pixel_surprise - running_mean_pixel) / running_std_pixel
        
        # Cập nhật trạng thái chia sẻ (Global state) dưới khóa Lock bảo vệ
        with state_lock:
            # Resize frame game trước khi lưu để stream mượt mà, tránh tốn băng thông
            current_frame = cv2.resize(frame_rgb, (320, 240))
            current_metrics = {
                "step": step_count,
                "action": action_val,
                "latent_surprise": latent_surprise,
                "pixel_surprise": pixel_surprise,
                "norm_latent": max(-3.0, min(10.0, norm_latent)), # Clip hiển thị biểu đồ đẹp hơn
                "norm_pixel": max(-3.0, min(10.0, norm_pixel)),
                "is_ood": is_currently_ood,
                "frameskip": env.frameskip
            }
            
        obs_processed = next_obs_processed
        step_count += 1
        
        if done:
            # Nếu game chết, giữ nguyên frameskip hiện tại và tiếp tục tập chơi ván mới
            current_fs = env.frameskip
            obs, info = env.reset()
            env.frameskip = current_fs
            obs_processed = preprocess_frame(obs)
            update_frame_stack(obs_processed)
            
        # Điều tiết tốc độ khung hình hiển thị (tương đương 30fps)
        time.sleep(0.03)

# ---- ĐỊNH NGHĨA CÁC ROUTE CỦA FLASK SERVER ----

@app.route('/')
def index():
    """Trả về giao diện Dashboard"""
    return render_template_string(HTML_TEMPLATE)

def gen_video_frames():
    """Tạo luồng JPEG từ frame game hiện tại để stream MJPEG"""
    global current_frame
    while True:
        frame_bytes = None
        with state_lock:
            if current_frame is not None:
                # Chuyển đổi từ RGB sang BGR để OpenCV mã hóa đúng màu sắc
                frame_bgr = cv2.cvtColor(current_frame, cv2.COLOR_RGB2BGR)
                _, jpeg = cv2.imencode('.jpg', frame_bgr)
                frame_bytes = jpeg.tobytes()
                
        if frame_bytes is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.04) # ~25 FPS

@app.route('/video_feed')
def video_feed():
    """Endpoint cung cấp video stream MJPEG"""
    return Response(gen_video_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_metrics_stream():
    """Tạo luồng Server-Sent Events gửi dữ liệu Metrics"""
    global current_metrics
    last_step = -1
    while True:
        data_to_send = None
        with state_lock:
            if current_metrics is not None and current_metrics["step"] != last_step:
                last_step = current_metrics["step"]
                data_to_send = json.dumps(current_metrics)
                
        if data_to_send is not None:
            yield f"data: {data_to_send}\n\n"
            
        time.sleep(0.04)

@app.route('/metrics')
def metrics():
    """Endpoint cung cấp metrics thời gian thực dạng SSE"""
    return Response(gen_metrics_stream(), mimetype='text/event-stream')

@app.route('/trigger_action', methods=['POST'])
def trigger_action():
    """Endpoint nhận lệnh điều khiển từ Giao diện Web"""
    global command_queue
    data = request.json
    action_type = data.get("action")
    
    if action_type == "reset":
        command_queue.append({"type": "reset"})
        return jsonify({"status": "success", "message": "Queue restart command"})
    elif action_type == "set_frameskip":
        fs_value = int(data.get("value", 4))
        command_queue.append({"type": "set_frameskip", "value": fs_value})
        return jsonify({"status": "success", "message": f"Queue change frameskip to {fs_value}"})
        
    return jsonify({"status": "error", "message": "Invalid command"}), 400

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive Web Dashboard for LeWM OOD Detection")
    parser.add_argument("--env", type=str, default="ALE/Pong-v5", help="Atari Environment ID")
    parser.add_argument("--port", type=int, default=5000, help="Web Server Port")
    parser.add_argument("--latent_dim", type=int, default=64, help="Latent Space Dimension")
    args = parser.parse_args()
    
    # Khởi chạy luồng giả lập game trong background thread
    sim_thread = threading.Thread(
        target=run_simulation, 
        args=(args.env, args.latent_dim),
        daemon=True
    )
    sim_thread.start()
    
    # Khởi chạy Flask Server trên cổng đã cấu hình
    print(f"\\n[Dashboard] Khởi động Flask Server tại http://127.0.0.1:{args.port}...")
    app.run(host='0.0.0.0', port=args.port, debug=False, threaded=True)
