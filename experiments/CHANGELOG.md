# Lịch sử phiên bản code (`defectfill_02`)

Mỗi phiên bản là một tag `defectfill_02-vN` trên branch `defectfill_02`. Baseline là branch `main` (thuật toán của tag `baseline-original` = `c825dff`). Kết quả từng run nằm trong `RESULTS.md` và `records/`. Đối chiếu công thức paper ↔ code: `formula/formula_mapping.md` (bản sao local, không có trên git) và hướng dẫn server, Phần F.

Paper (hazelnut, trung bình mọi loại defect, 1000 ảnh): KID 1.13, IC-LPIPS 0.31.

## v3 — thu mask về latent bằng max-pool, ghép background ở không gian pixel (tag `defectfill_02-v3`)

**Vấn đề** (README baseline: "slender mask branches sometimes not fully converted into defects", "slight yellow-green halos"; đo bằng `tools/mask_fill.py` trên 100 ảnh):

| | baseline | v2 |
|---|---|---|
| Nét mảnh của mask được vẽ lại (`fill_thin`) | 6.2% | 4.7% |
| Phần dày của mask được vẽ lại (`fill_thick`) | 46.1% | 34.3% |
| Nền xa mask bị đổi (`drift_far`, 0–255) | 2.25 | 2.94 |

**Nguyên nhân:**

1. Mask 512×512 được thu về latent 64×64 bằng `F.interpolate(..., mode="nearest")`, tức chỉ lấy 1 pixel trong mỗi ô 8×8. Với 12 mask held-out của hazelnut/hole, mask latent chỉ giữ 67–89% pixel defect, và trung bình **39% pixel của nét mảnh** (mask `006`: 19%). Pipeline diffusers cũng thu mask theo cách này. Hệ quả: nét mảnh không nằm trong kênh mask của UNet, không được tính trong $\mathcal L_{def}$ hay đích của $\mathcal L_{attn}$, và lúc inference bị bước giữ background ghi đè.
2. Ảnh cuối cùng là bản decode qua VAE của **toàn bộ** latent, nên cả phần nền là ảnh tái tạo của VAE chứ không phải ảnh gốc. Paper chỉ yêu cầu giữ nguyên nền.

**Thay đổi:**

- `model.prepare_inpaint_masks(mask, latent_size, mode)`. Với `maxpool` (mặc định): một ô latent là mask khi có bất kỳ pixel defect nào trong ô, và mask pixel được căn theo ô, để ảnh nền $B$, kênh mask và loss cùng một vùng. `nearest` giữ hành vi cũ.
  - `train.py`: dùng cho nhánh defect, nhánh object, $M'$ và đích của $\mathcal L_{attn}$. Thêm `--latent_mask_mode`, có trong fingerprint resume.
  - `model.generate` / `inference.py`: dùng cho mask đưa vào pipeline và mask của bước giữ background. Thêm `--latent_mask_mode`.
- `model.composite_background`: sau khi decode, giữ pixel gốc ngoài vùng inpaint, mép làm mềm 2 px. `inference.py --composite on|off`, mặc định `on`.
- `tools/mask_fill.py` (đo), `tools/record_run.py` và thư mục `experiments/` (sổ thí nghiệm).

**Kiểm chứng trước khi train** (PC, checkpoint v2, chỉ đổi inference, 4 cặp có nét mảnh): `fill_thin` tăng 0% → 14%, 22% → 33%, 4% → 17%; `fill_thick` tăng ở mọi cặp; `drift_far` = 0. Trên ảnh, vết nứt mảnh được vẽ dọc theo mask.

**Ghi chú:** paper không nói cách thu mask về latent hay có ghép pixel không. Đây là lựa chọn của repo, theo đúng ý paper: defect phải theo đúng mask, còn nền phải giữ nguyên. Nhãn (mask lưu cùng ảnh sinh) vẫn là mask gốc; vùng sinh lớn hơn mask tối đa 7 px theo lưới 8×8.

Run: `20261015_v3_hazelnut_hole_100img` (train + inference v3), `20261015_v3infer_v2ckpt_hazelnut_hole_100img` (checkpoint v2 + inference v3, để tách hiệu quả của phần inference).

## v2 — loss scaling cho backward fp16 (tag `defectfill_02-v2` = `784e8cd`)

**Vấn đề:** v1 có KID×1000 = 144 so với −1.5 của baseline. Vùng mask bị nhiễu màu, và $\mathcal L_{def}$, $\mathcal L_{obj}$ **tăng** trong lúc train.

**Nguyên nhân:** UNet chạy fp16 không có loss scaling. Loss chuẩn hóa theo paper (L1) nhỏ hơn khoảng 170 lần, nên gradient underflow: cos với gradient đúng chỉ 0.08–0.79, và ở t=900 chỉ 19% gradient khác 0.

**Thay đổi:** `torch.amp.GradScaler` (lưu trong checkpoint, ghi `LossScale` vào TensorBoard, có `loss_scaling` trong fingerprint resume).

**Kết quả:** loss giảm trở lại, ảnh sạch, tương đương baseline ở 100 ảnh. Run: `20261014_v2_hazelnut_hole_100img`.

## Merge `main` (`c64d3b5`) — không đổi thuật toán

Tự xóa checkpoint trung gian khi train xong (`--keep_intermediate_checkpoints` để giữ); README dùng 100 ảnh khi code chưa chốt.

## v1 — sửa theo công thức paper (tag `defectfill_02-v1` = `296ccbc`)

So với baseline (chi tiết trong `formula/formula_mapping.md`, mục 3):

- L1: MSE có mask chia cho hằng số (Eq. 5/7), thay vì chia cho diện tích mask.
- L2: $M'$ nằm trong chuẩn bình phương, nên background có trọng số $\alpha^2$.
- L4: một hàm `generate_random_box_mask` cho $M_{rand}$; bỏ `adjusted_mask` thừa.
- L6: Low-Fidelity Selection dùng mask nhị phân.
- L7: bỏ `set_timesteps(30)` thừa.
- L8: tâm cluster của IC-LPIPS là ảnh train (Ojha et al.).

**Kết quả:** hỏng do gradient underflow (sửa ở v2). Run: `20261012_v1_hazelnut_hole_1000img`.

## baseline (`main`, thuật toán `c825dff`)

Code unofficial gốc, đã được Codex sửa theo protocol paper (chia 1/3, sinh full-view, resume). Điểm yếu tự nêu trong README: lỗ tròn/đen na ná nhau, viền vàng-xanh, nét mảnh của mask không thành defect; KID tốt nhưng IC-LPIPS thấp. Run: `20261012_baseline_hazelnut_hole_1000img`, `20261014_baseline_hazelnut_hole_100img`.
