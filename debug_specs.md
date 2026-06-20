# SPECS ĐIỀU TRA — Debug evaluate_ood.py (Pong + Procgen results)

Bối cảnh: Đây là pipeline đánh giá OOD Dynamics Detection cho nghiên cứu
LeWM (JEPA world model) hướng tới publish NeurIPS/ICLR/ICML. Hai hình kết quả
(pong_results.png, procgen_results.png) cho ra số liệu NGƯỢC với kỳ vọng
(Baseline AUROC > LeWM AUROC), và có dấu hiệu bug rõ ràng. Cần điều tra
TRƯỚC KHI viết lại paper, vì nếu code sai thì mọi số liệu đều vô giá trị.

Hãy trả lời TỪNG CÂU HỎI dưới đây bằng cách đọc trực tiếp code trong project,
KHÔNG đoán, paste đoạn code liên quan làm bằng chứng cho mỗi câu trả lời.

---

## CÂU HỎI 1 — Xác định phiên bản evaluate_ood.py đang dùng

Mở file `evaluate_ood.py` (hoặc file đã tạo ra 2 ảnh pong_results.png /
procgen_results.png). Paste toàn bộ nội dung file này.

Đặc biệt tìm và paste riêng phần code:

- Hàm/đoạn code vẽ panel "Return Degradation" (subplot thứ 3)
- Hàm/đoạn code tính "Corr(LeWM)" hiển thị trong legend
- Hàm/đoạn code tính "Smoothed Reward"

---

## CÂU HỎI 2 — Bug Corr(LeWM) = 0.00 tuyệt đối

Trong CẢ 3 trường hợp (Pong fs→2, Procgen fs→2, Procgen fs→8), giá trị
`Corr(LeWM)` hiển thị đúng `0.00` không sai một số lẻ nào. Đây là dấu hiệu
bug, không phải trùng hợp.

Hãy kiểm tra:

1. Hàm tính correlation dùng `np.corrcoef()` hay `scipy.stats.pearsonr()`
   hay tự viết thủ công?
2. In ra (print) giá trị thô của 2 mảng đưa vào correlation (latent_surprise
   array và smoothed_reward array) — kiểm tra xem một trong hai mảng có
   std = 0 (hằng số, không đổi) hay không. Nếu std=0 thì corrcoef sẽ trả về
   NaN, và có khả năng code đang dùng `np.nan_to_num()` hoặc làm tròn khiến
   NaN hiển thị thành "0.00".
3. Hãy paste output thực tế của:
   ```python
   print("Latent surprise std:", latent_surprise_array.std())
   print("Smoothed reward std:", smoothed_reward_array.std())
   print("Raw corrcoef:", np.corrcoef(latent_surprise_array, smoothed_reward_array))
   ```
4. Kiểm tra xem correlation được tính trên TOÀN BỘ trajectory hay chỉ trên
   đoạn "OOD phase" (sau ood_step) như paper mô tả (main.tex viết: "Pearson
   correlation coefficient between the Latent Surprise and the Smoothed
   Reward during the OOD phase"). Nếu đang tính trên toàn bộ thay vì chỉ
   OOD phase, đây là một bug khác cần sửa.

---

## CÂU HỎI 3 — Return Degradation giảm TRƯỚC ood_step (rất quan trọng)

Trong cả 3 hình, đường "Smoothed Reward" rơi xuống gần 0 RẤT SỚM (khoảng
step 60-130/230), TRƯỚC KHI OOD được trigger (step 100/500). Điều này vô lý
nếu giả thuyết là "OOD dynamics gây ra return degradation".

Hãy kiểm tra và trả lời:

1. Đoạn code tính "Smoothed Reward" — nó dùng EMA của `reward` mỗi step,
   hay dùng return tích lũy (cumulative return) của episode?
   Paste công thức/code chính xác.
2. Trong Pong, mỗi điểm thua/thắng (point scored) trả `reward=-1` hoặc
   `reward=+1` rồi MÔI TRƯỜNG TỰ RESET liên tục trong cùng 1 "match" (21
   điểm). Việc agent dùng RANDOM ACTION (không có PPO checkpoint) có thể
   khiến agent thua điểm rất nhanh (trong ~60-100 step) một cách HOÀN TOÀN
   BÌNH THƯỜNG, không liên quan gì đến OOD. Hãy kiểm tra:
   - Có file `models/ppo_ALE_Pong_v5.zip` tồn tại trong thư mục `models/`
     không? Hãy chạy `ls -la models/` và paste kết quả.
   - Có file `models/ppo_procgen_*.zip` (PPO cho Procgen) tồn tại không?
   - Nếu PPO model KHÔNG tồn tại, code có in cảnh báo "Dùng Random Agent"
     ra console khi chạy không? Paste console log đầy đủ của lần chạy gần
     nhất nếu còn lưu (terminal history, log file, hoặc chạy lại và xem).
3. Khi `done=True` (agent thua điểm/game over) giữa episode, code có
   gọi `env.reset()` và việc reset này CÓ VÔ TÌNH RESET `frameskip` về
   `initial_frameskip` (tức làm mất trạng thái OOD đã trigger) hay không?
   Đối chiếu với logic trong `atari_ood_wrapper.py`:
   ```python
   def reset(self, **kwargs):
       obs, info = self.env.reset(**kwargs)
       self.obs_buffer[0] = obs
       self.obs_buffer[1] = obs
       return obs, info
   ```
   Lưu ý: hàm `reset()` ở trên KHÔNG đụng tới `self.frameskip`, nên về lý
   thuyết frameskip vẫn giữ nguyên sau reset. Nhưng hãy xác nhận lại bằng
   cách thêm `print(f"step={step}, frameskip={env.frameskip}, is_ood={env.is_ood}")`
   ngay sau mỗi `env.reset()` trong vòng lặp evaluation, chạy lại, và paste
   log để xác nhận chắc chắn.

---

## CÂU HỎI 4 — Đối chiếu file checkpoint thực tế đã dùng

1. Chạy lệnh sau và paste kết quả:
   ```bash
   ls -la models/
   ls -la datasets/
   ```
2. Với mỗi file `.pth` trong `models/`, paste thời gian tạo file (`ls -la`
   đã có) — để xác nhận checkpoint LeWM dùng để eval Pong có phải được
   train ĐÚNG trên Pong (không phải nhầm checkpoint của môi trường khác).
3. Paste output của lệnh chạy training cuối cùng cho LeWM Pong (log
   training loss qua các epoch) nếu còn lưu — để xác nhận loss đã hội tụ
   tốt, không phải model train dở/under-fit.

---

## CÂU HỎI 5 — Cấu hình thực tế đã dùng để chạy ra 2 ảnh

Paste chính xác lệnh (command) đã dùng để chạy ra `pong_results.png` và
`procgen_results.png`. Ví dụ cần biết:

- `--steps`, `--ood_step`, `--frameskip_variants`, `--seeds` là bao nhiêu?
- Ảnh Pong chỉ thấy 1 frameskip variant (fs→2) trong khi lệnh gốc tôi viết
  default là `[2, 8]` — có phải bạn đã sửa `--frameskip_variants 2` khi
  chạy Pong? Hay code đã bị sửa để chỉ chạy 1 variant cho Pong?
- `--seeds` dùng bao nhiêu? Ảnh hiện AUROC có "±0.000" (std=0), nghĩa là
  hoặc seeds=1, hoặc bug khiến mọi seed ra kết quả giống nhau y hệt. Hãy
  xác nhận giá trị `--seeds` thực tế đã dùng.

---

## CÂU HỎI 6 — Procgen: kiểm tra category môi trường

Paper hiện tại (main.tex) viết môi trường `procgen:coinrun-v0`, nhưng
title trong ảnh `procgen_results.png` hiển thị `procgen:procgen-coinrun-v0`
(có vẻ lặp từ "procgen" 2 lần). Hãy xác nhận:

1. `env_id` chính xác đang dùng cho Procgen là gì? Paste dòng code khởi
   tạo env Procgen.
2. Action space của Procgen CoinRun có khớp với `action_dim` đang được
   truyền vào `LeWorldModel` không? (CoinRun có 15 actions theo
   `procgen` gốc, KHÔNG phải 6 như Pong — nếu code đang hardcode
   `action_dim=6` cho Procgen thì đây là bug nghiêm trọng làm hỏng toàn
   bộ action embedding).
3. Paste dòng code xác định `action_dim` khi train/eval Procgen.

---

## CÂU HỎI 7 — [MỚI] Procgen kết quả KHÔNG ỔN ĐỊNH qua seed (rất nghiêm trọng)

Đã chạy `evaluate_ood.py` 2 lần trên Procgen CoinRun với CÙNG một phiên bản
code (không sửa gì), chỉ khác random seed. Kết quả dao động RẤT MẠNH:

| Variant   | Lần chạy 1 (cũ) | Lần chạy 2 (mới) | Chênh lệch |
| --------- | --------------- | ---------------- | ---------- |
| fs→2 LeWM | AUROC = 0.528   | AUROC = 0.388    | 0.140      |
| fs→2 Base | AUROC = 0.580   | AUROC = 0.133    | **0.447**  |
| fs→8 LeWM | AUROC = 0.445   | AUROC = 0.501    | 0.056      |
| fs→8 Base | AUROC = 0.506   | AUROC = 0.510    | 0.004      |

Baseline AUROC=0.133 ở lần chạy 2 (tệ hơn random một cách cực đoan, gần như
"anti-predictive hoàn hảo") là dấu hiệu cực kỳ bất thường — không giống biến
động tự nhiên do random seed thông thường.

Ngoài ra, trong CẢ HAI lần chạy, legend ROC curve luôn hiện `±0.000`
(ví dụ "LeWM AUROC=0.528±0.000"), nghĩa là mỗi lần chạy chỉ dùng **1 seed
duy nhất** (`--seeds 1`), KHÔNG phải multi-seed averaging như thiết kế gốc
yêu cầu (`run_multi_seed` lấy mean±std qua N seeds độc lập TRONG CÙNG một
lần gọi script, không phải chạy lại script nhiều lần riêng rẽ).

Hãy kiểm tra và trả lời:

1. Paste chính xác lệnh (command) đã dùng để chạy ra `procgen_results.png`
   và `procgen_results_new.png`. Cụ thể giá trị `--seeds` là bao nhiêu?
2. Nếu `--seeds` đang để mặc định hoặc =1, đây CHÍNH LÀ nguyên nhân
   `±0.000` — không phải bug code, mà là **chưa chạy đủ seeds**. Cần chạy
   lại với `--seeds 5` (hoặc tối thiểu 3) để có mean±std đáng tin cậy
   cho paper.
3. Việc Baseline AUROC nhảy từ 0.580 → 0.133 chỉ vì đổi seed cho thấy
   PixelPredictor baseline có thể đang **chưa train đủ converge** (loss
   chưa ổn định), khiến hành vi dự đoán rất nhạy với episode/seed cụ thể.
   Hãy paste training loss curve (giá trị loss qua từng epoch) của
   `baseline` trên Procgen từ log training để xác nhận model đã converge
   hay chưa.
4. Procgen CoinRun là môi trường procedurally generated — MỖI EPISODE có
   level/background hoàn toàn khác nhau (theo `seed`). Nếu code đang dùng
   `env.action_space.seed(seed)` để set seed nhưng KHÔNG set seed cho chính
   procedural generation của Procgen (thường cần qua `gym.make(..., start_level=seed, num_levels=1)`
   hoặc tương đương), thì "seed khác" có thể đang vô tình tạo ra một LEVEL
   HOÀN TOÀN KHÁC mỗi lần chạy — giải thích tại sao biến động AUROC lớn đến
   vậy. Hãy paste dòng code khởi tạo môi trường Procgen và xác nhận cách
   set seed/level đang được dùng.

---

## TÓM TẮT — Output mong đợi

Sau khi trả lời 7 câu hỏi trên (câu 1-6 cũ + câu 7 mới) với code thật +
log thật (không suy đoán), gửi lại toàn bộ cho Claude (Anthropic) để xác
định:

- (a) Đây là bug code cần sửa trước khi có số liệu hợp lệ, hay
- (b) Đây là finding thật (LeWM thực sự kém hơn Baseline trên Pong/Procgen,
  và kết quả Procgen có variance cao tự nhiên do procedural generation)
  cần diễn giải lại trong paper với multi-seed averaging đầy đủ.
