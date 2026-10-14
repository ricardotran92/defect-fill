# experiments/ — sổ thí nghiệm DefectFill

Thư mục này lưu **mọi thứ cần để truy vết và so sánh các run**, ở dạng file nhỏ được git theo dõi. Ảnh sinh ra và checkpoint vẫn nằm trên server (`runs/`, bị `.gitignore`).

| Đường dẫn | Nội dung | Ai cập nhật |
|---|---|---|
| `CHANGELOG.md` | Lịch sử phiên bản code: commit/tag, thay đổi, lý do, bằng chứng, run liên quan | Sửa tay khi ra phiên bản mới |
| `registry.json` | Danh sách run đã ghi: `run_id`, phiên bản, commit, đường dẫn bản ghi, dấu vân tay output | `tools/record_run.py` |
| `records/<run_id>/run.json` | Bản ghi đầy đủ một run: train config, tóm tắt loss (TensorBoard), cấu hình inference, điểm LFS, metrics (tự chấm + chấm chéo), hiệu năng, mức vẽ kín mask, nơi lưu artifact | `tools/record_run.py` |
| `RESULTS.md` | Bảng so sánh mọi run, sinh lại từ `registry.json` | `tools/record_run.py` (không sửa tay) |
| `scripts/` | Script chạy trên server (`run_pipeline.sh`, `run_inference_only.sh`, `run_xeval.sh`) và script tải kết quả về PC (`fetch_run.sh`) | Sửa qua git |

## Quy ước

- **`run_id`** = `YYYYMMDD_<phiên bản>_<object>_<defect>_<N>img[_ghi chú]`, chữ thường, nối bằng `_`. Ví dụ `20261014_v2_hazelnut_hole_100img`. Ngày là ngày run chạy xong.
- **Phiên bản** = `baseline` (branch `main`) hoặc `vN` (tag `defectfill_02-vN`). Run chỉ dùng inference mới trên checkpoint cũ đặt tên kiểu `v3infer_v2ckpt` và ghi `--reuses_checkpoint_of`.
- **Thư mục output trên server** theo phiên bản: `runs/<object>_<defect>_paper_2000step_<vN>` và `runs/<object>_<defect>_paper_generated_<vN>_<N>`. Không dùng lại thư mục của phiên bản khác.
- **Chỉ chạy code đã commit.** Các script server từ chối chạy khi worktree có thay đổi chưa commit, để `commit` trong bản ghi luôn đúng.
- **Không trùng lặp:** `record_run.py` từ chối ghi một `run_id` đã có (trừ khi có `--update`), và từ chối ghi cùng một output dưới hai `run_id` (so dấu vân tay của `inference_log.json`).

## Quy trình một run

```bash
# 1. PC: sửa code -> pytest -> commit -> tag -> push
git tag defectfill_02-v3 && git push origin defectfill_02 defectfill_02-v3

# 2. Server: pull, rồi chạy trong tmux (xem hướng dẫn server, Phần F)
VERSION=v3 bash experiments/scripts/run_pipeline.sh            # train -> 100 ảnh -> evaluate
GEN_NAME=v3 RUN_VERSION=v3 bash experiments/scripts/run_xeval.sh  # chấm chéo bằng evaluate.py baseline

# 3. PC (Git Bash): tải phần nhẹ của run về worktree tương ứng
bash experiments/scripts/fetch_run.sh defectfill_02 hazelnut_hole_paper_2000step_v3 hazelnut_hole_paper_generated_v3_100
bash experiments/scripts/fetch_run.sh defect-fill - hazelnut_hole_paper_generated_100   # nếu cần file chấm chéo mới

# 4. PC: ghi run vào sổ, rồi commit thư mục experiments/
python tools/record_run.py --run_id 20261015_v3_hazelnut_hole_100img --version v3 --commit <hash> \
  --train_dir runs/hazelnut_hole_paper_2000step_v3 --generated_dir runs/hazelnut_hole_paper_generated_v3_100 \
  --metrics self runs/paper_protocol_metrics_server.csv defectfill_02_v3_100img \
  --metrics xeval_baseline_evaluator ../defect-fill/runs/compare/xeval_100.csv v3_out__baseline_eval_100img \
  --perf_csv runs/perf/perf_summary_server.csv --server_root kindred:~/projects/defectfill_02 --notes "..."
git add experiments && git commit -m "experiments: record 20261015_v3_hazelnut_hole_100img"
```

## Đọc kết quả

- Chỉ so các run có **cùng số ảnh**. Chỉ so IC-LPIPS khi **cùng tâm cluster** (`ref`: `train` = `evaluate.py` mới, `heldout` = `evaluate.py` baseline).
- KID chỉ có 12 ảnh thật cho hazelnut/hole, nên độ lệch chuẩn khoảng ±4 (×10⁻³), lớn hơn chênh lệch giữa các run tốt.
- `fill_thick`, `fill_thin`, `drift_far` (`tools/mask_fill.py`) đo hai điểm yếu nêu trong README baseline: nét mảnh của mask không được vẽ, và nền ngoài mask bị đổi màu. Phải đọc kèm KID và ảnh: run hỏng v1 có `fill_*` rất cao vì vùng mask bị phủ nhiễu.
- Số của paper cho hazelnut (trung bình các loại defect, 1000 ảnh): KID 1.13, IC-LPIPS 0.31 (Table 1, arXiv 2503.13985v2).
