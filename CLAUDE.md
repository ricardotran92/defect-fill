# DefectFill — code mới (`defectfill_02`): hướng dẫn cho agent

Trả lời người dùng bằng **tiếng Việt**. Đây là worktree `defectfill_02` của repo `ricardotran92/defect-fill`; baseline (code gốc) là worktree `D:\repos\defect-fill`, branch `main`, **không sửa thuật toán ở baseline**.

## Quy tắc bắt buộc

- **Mọi việc tính toán chỉ chạy trên server** `ricardo@140.123.104.126` (ssh key có sẵn), **GPU 2** (`CUDA_VISIBLE_DEVICES=2` / `--gpu 2`). Không chạy train/inference/model SD2 trên PC. PC chỉ: sửa code, `pytest` (CPU), git, xem/ghép ảnh đã tải về.
- Server: `source /opt/miniconda3/etc/profile.d/conda.sh && conda activate ~/envs/defectfill`; code mới `~/projects/defectfill_02`, baseline `~/projects/defect-fill` (file code khóa ghi: mở `chmod u+w`, pull, khóa lại `chmod a-w *.py tests/*.py tools/*.py`). Dữ liệu `~/shared/datasets/mvtec_ad`, model `~/shared/models/sd2_inpainting`.
- **Chỉ chạy code đã commit + tag** (`defectfill_02-vN`); script server từ chối chạy khi có thay đổi chưa commit. Chạy dài trong tmux, mỗi lúc một job trên GPU 2. Không `git pull` khi job đang chạy.
- **Mọi run phải ghi vào sổ** `experiments/` bằng `tools/record_run.py` (chặn trùng run_id/output), cập nhật `experiments/CHANGELOG.md`, rồi commit + push. Chẩn đoán (không sinh ảnh) ghi vào `experiments/diagnostics/`.
- Repo dùng line ending lẫn lộn (CRLF/LF): sửa file bằng Edit hoặc patch giữ line ending; script `.sh` phải là LF.

## Tra cứu (mở khi cần, đừng đọc hết)

| Cần | Ở đâu |
|---|---|
| Kết quả mọi run (bảng) | `experiments/RESULTS.md` |
| Lịch sử phiên bản: vấn đề → nguyên nhân → thay đổi → kết quả | `experiments/CHANGELOG.md` |
| Chẩn đoán (attention, split) | `experiments/diagnostics/README.md` |
| Quy trình chạy / tải về / ghi sổ | `experiments/README.md`, `experiments/scripts/*.sh` |
| Nhật ký đầy đủ (F0–F13) và hướng dẫn server | `C:\Users\Ricardo\Box\ccu\lab\server\Nhat_ky_phat_trien_defectfill_02.md`, `Huong_dan_DefectFill_kindred_v5.md` |
| Thông số paper | `D:\repos\defect-fill\source_materials\INDEX.md` → `paper-parameters.md` (đừng mở PDF trừ khi cần) |

## Trạng thái (17/10/2026)

- Code mới nhất: tag `defectfill_02-v7` (`5e55f3d`); sổ thí nghiệm mới nhất `0c3dc74`. 68 test pass.
- **Cấu hình nên dùng = v6**: `--masked_loss_norm mask --attention_loss_norm region --latent_mask_mode maxpool --thin_weight 1.0` (đều là mặc định trừ `thin_weight`), `GradScaler`, ảnh nền theo mask gốc, ghép background, guidance 2.0, LFS mask nhị phân.
- Đánh giá đang dùng: 100 ảnh, kế hoạch ưu tiên 3:1 trên 5 mask nét mảnh `006,010,011,014,016` (`GEN_SUFFIX=p`, `--priority_masks ... --priority_ratio 3`). KID chỉ có 12 ảnh thật nên nhiễu ±4–5; so bằng ảnh và `tools/mask_fill.py` (`fill_thin`, `fill_thick`, `drift_far`).
- **Mục tiêu người dùng:** vết nứt rõ ở nét mảnh của mask như Hình 6 / S9 paper.
- **Kết luận đã có:** lỗ đẹp nhất ở v6; attention của `sks` đã phủ nét mảnh (chẩn đoán attention); tăng trọng số phần mảnh (v7) không giúp; **nút thắt là dữ liệu tham chiếu** — split mặc định seed 42 cho 6 ảnh train gần như không có vết nứt (nét mảnh 2.8%); với `--split_seed 113` (train có vết nứt, chẩn đoán, lệch protocol) model lần đầu vẽ vết nứt tối mảnh (mask 016 `fill_thin` 20.4% vs 3.3%).
- Code chính thức DefectFill không công bố. Gần nhất có code: MAGIC (SD2 inpainting + DreamBooth, github Jaeihk/MAGIC-Anomaly-generation, ví dụ không có vết nứt mảnh); DualAnoDiff (không inpainting, có vết nứt hazelnut rõ nhưng đổi nền); AnomalyDiffusion (có bộ ảnh MVTec sinh sẵn).

## Bước tiếp theo đang chờ người dùng chọn

1. Train v6 trên `--defect_type crack` (MVTec hazelnut có loại `crack` riêng) — kiểm tra vết nứt rõ như Hình S9.
2. Tải bộ ảnh sinh sẵn của AnomalyDiffusion (hazelnut) để xem vết nứt (không tốn GPU).
3. Chạy MAGIC bằng code gốc, cùng split 42 + 5 mask ưu tiên (so sánh có kiểm soát).
4. Báo cáo theo protocol: v6 với nhiều `split_seed` (42, 113, 166) lấy trung bình; hoặc chia phân tầng theo tỉ lệ nét mảnh (lệch protocol, phải ghi rõ).

## Lệnh mẫu (server, trong tmux)

```bash
cd ~/projects/defectfill_02 && P="--priority_masks 006_mask.png,010_mask.png,011_mask.png,014_mask.png,016_mask.png --priority_ratio 3"
VERSION=v8 NIMG=100 GEN_SUFFIX=p TRAIN_ARGS="..." INFER_ARGS="$P" bash experiments/scripts/run_pipeline.sh 2>&1 | tee -a runs/pipeline_v8.log
GEN_NAME=v8 RUN_VERSION=v8 NIMG=100 GEN_SUFFIX=p bash experiments/scripts/run_xeval.sh 2>&1 | tee -a runs/pipeline_v8.log
```

PC (Git Bash) sau khi xong: `bash experiments/scripts/fetch_run.sh defectfill_02 <train_dir> <generated_dir>` → `python tools/record_run.py --run_id YYYYMMDD_<ver>_hazelnut_hole_100img_priority ...` (xem `experiments/README.md`). Python có CUDA ở PC: `D:\repos\defect-fill\.venv\Scripts\python.exe` (chỉ dùng cho phân tích CPU).
