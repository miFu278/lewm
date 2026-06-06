3.2. Đề tài B - an toàn học thuật: OOD dynamics detection trong Atari/Procgen
Tên đề xuất: Can Lightweight JEPA-based World Models Detect Out-of-Distribution Dynamics in Atari and Procgen Games?
Dataset/benchmark chính: Atari/ALE và Procgen Benchmark.
Giả thuyết nghiên cứu: LeWM có thể nhận biết thay đổi dynamics qua latent surprise, ngay cả khi thay đổi chưa thể hiện rõ trong một frame đơn lẻ.
Cách làm: train trên môi trường chuẩn; tạo biến thể OOD bằng cách đổi tốc độ, gravity, frame skip, enemy behavior hoặc transition rule.
Metric: OOD AUROC, F1, return degradation, planning success, calibration error.
Điểm mạnh: benchmark quen thuộc trong RL/world model; reviewer AI dễ hiểu; ít phụ thuộc vào dữ liệu tự tạo.