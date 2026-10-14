# Kết quả các run (tự sinh bởi `tools/record_run.py`, không sửa tay)

Chỉ so các dòng có cùng số ảnh, và chỉ so IC-LPIPS khi cùng tâm cluster (`ref`). `fill_thin`/`fill_thick`: tỉ lệ pixel trong phần mảnh/dày của mask được vẽ lại; `drift_far`: thay đổi trung bình (0–255) ở nền xa mask, lý tưởng là 0. Định nghĩa trong `tools/mask_fill.py`.

| run_id | phiên bản | commit | ảnh | KID×1000 (tự chấm) | IC-LPIPS (tự chấm, ref) | chấm chéo | fill_thick | fill_thin | drift_far | L_def đầu → cuối |
|---|---|---|---|---|---|---|---|---|---|---|
| [20261012_baseline_hazelnut_hole_1000img](records/20261012_baseline_hazelnut_hole_1000img/run.json) | baseline | `c825dff` | 1000 | -1.48 (±4.39) | 0.181 (heldout) | — | 47.3% | 5.6% | 2.27 | 1.1635 → 0.5237 |
| [20261012_v1_hazelnut_hole_1000img](records/20261012_v1_hazelnut_hole_1000img/run.json) | v1 | `296ccbc` | 1000 | 144.10 (±17.45) | 0.319 (train) | — | 97.7% | 82.5% | 15.35 | 0.0052 → 0.0085 |
| [20261014_baseline_hazelnut_hole_100img](records/20261014_baseline_hazelnut_hole_100img/run.json) | baseline | `751bde5` | 100 | -2.21 (±4.16) | 0.242 (heldout) | xeval_new_evaluator: KID -2.21, IC 0.255 (train) | 46.1% | 6.2% | 2.25 | — (dùng checkpoint có sẵn) |
| [20261014_v2_hazelnut_hole_100img](records/20261014_v2_hazelnut_hole_100img/run.json) | v2 | `784e8cd` | 100 | -0.03 (±4.09) | 0.254 (train) | xeval_baseline_evaluator: KID -0.03, IC 0.215 (heldout) | 34.3% | 4.7% | 2.94 | 0.0037 → 0.0018 |
| [20261014_v3_hazelnut_hole_100img](records/20261014_v3_hazelnut_hole_100img/run.json) | v3 | `ab2144d` | 100 | 0.43 (±4.58) | 0.259 (train) | xeval_baseline_evaluator: KID 0.43, IC 0.201 (heldout) | 50.2% | 28.6% | 0.00 | 0.0049 → 0.0024 |
| [20261014_v3infer_v2ckpt_hazelnut_hole_100img](records/20261014_v3infer_v2ckpt_hazelnut_hole_100img/run.json) | v3 | `ab2144d` | 100 | -0.43 (±4.50) | 0.259 (train) | xeval_baseline_evaluator: KID -0.43, IC 0.221 (heldout) | 50.6% | 24.1% | 0.00 | — (dùng checkpoint có sẵn) |
