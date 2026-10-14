# Lịch sử phiên bản code (`defectfill_02`)

Mỗi phiên bản là một tag `defectfill_02-vN` trên branch `defectfill_02`. Baseline là branch `main` (thuật toán của tag `baseline-original` = `c825dff`). Kết quả từng run nằm trong `RESULTS.md` và `records/`. Đối chiếu công thức paper ↔ code: `formula/formula_mapping.md` (bản sao local, không có trên git) và hướng dẫn server, Phần F.

Paper (hazelnut, trung bình mọi loại defect, 1000 ảnh): KID 1.13, IC-LPIPS 0.31.

## v4 — giữ max-pool cho mask latent, nhưng ảnh nền và đường ghép dùng mask gốc (tag `defectfill_02-v4`)

**Vấn đề của v3** (run `20261014_v3_hazelnut_hole_100img`): `fill_thin` tăng lên 28.6% và `drift_far` = 0 như mong đợi. Tuy nhiên lỗ và đường ghép có **hình khối vuông theo lưới 8×8**.

**Nguyên nhân:** v3 dùng mask căn theo ô latent cho cả hai việc sau:
- Ảnh nền $B=(1-M)\odot I$ bị khoét cả ô 8×8, nên model học vẽ khối vuông.
- Đường ghép pixel đi theo mép ô.

**Thay đổi:**

- `model.mask_to_latent` (thay `prepare_inpaint_masks`) chỉ tạo mask latent (max-pool). Mask này dùng cho kênh mask của UNet, cho loss, cho đích của $\mathcal L_{attn}$, và cho bước giữ background.
- Ảnh nền $B$ dùng **mask gốc độ phân giải đầy đủ**, đúng Eq. 4 của paper, ở cả train và inference. Khi inference, ảnh nền được truyền qua `masked_image_latents` (pipeline tự encode và không khoét lại). `mask_image` truyền vào là mask căn theo ô, để kênh mask của pipeline đúng bằng mask max-pool.
- `composite_background` giữ pixel sinh ra trên mask gốc nới rộng 3 px, mép mềm 2 px. Đường ghép đi theo hình defect.
- `train.py` có thêm `background_mask = "pixel"` trong fingerprint resume.

**Kiểm chứng trên PC** (checkpoint v2, cùng cặp ảnh/mask và seed với run ablation `v3infer_v2ckpt` trên server, ảnh 1, 3, 13): hết khối vuông, lỗ tròn, và nét mảnh được vẽ thành vết nứt liền mạch dọc theo mask.

**Kết quả** (100 ảnh, run `20261014_v4_hazelnut_hole_100img`):

| | baseline | v3 | v4 |
|---|---|---|---|
| `fill_thin` | 6.2% | 28.6% | **22.1%** |
| `fill_thick` | 46.1% | 50.2% | 43.7% |
| `drift_far` | 2.25 | 0 | **0** |
| KID×1000 (±) | −2.21 (4.16) | 0.43 (4.58) | 0.77 (4.74) |
| IC-LPIPS, tâm = ảnh train | 0.255 | 0.259 | 0.254 |
| IC-LPIPS, tâm = ảnh held-out | 0.242 | 0.201 | **0.241** |

- Hết khối vuông. Lỗ tròn, mép tự nhiên, bên trong có chiều sâu thay vì chỉ là chấm đen như baseline. IC-LPIPS tâm = ảnh held-out trở lại mức baseline sau khi tụt ở v3.
- `fill_thin` thấp hơn v3 vì v3 vẽ lại cả ô 8×8 quanh nét mảnh, nên đo được nhiều pixel đổi màu hơn. Tuy vậy v4 vẫn gấp khoảng 3.5 lần baseline.
- **Còn yếu:** nét mảnh mới chỉ là vệt sáng nhạt dọc theo mask, chưa giống vết nứt tối như defect thật. KID và IC-LPIPS vẫn ngang baseline trong vùng nhiễu, nên chưa có bằng chứng định lượng là v4 gần paper hơn (paper: KID 1.13, IC-LPIPS 0.31 cho hazelnut).

## v3 — thu mask về latent bằng max-pool, ghép background ở không gian pixel (tag `defectfill_02-v3` = `ab2144d`)

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

**Kết quả** (100 ảnh): `fill_thin` 28.6% (baseline 6.2%), `fill_thick` 50.2%, `drift_far` 0. KID×1000 = 0.43 (±4.6) và IC-LPIPS = 0.259 (tâm = ảnh train), ngang baseline trong vùng nhiễu. **Nhược điểm mới:** lỗ và đường ghép bị khối vuông theo lưới 8×8; sửa ở v4.

Ablation checkpoint v2 + inference v3: `fill_thin` 24.1%. Như vậy phần lớn cải thiện đến từ inference; train với max-pool thêm khoảng 4 điểm.

Run: `20261014_v3_hazelnut_hole_100img` (train + inference v3), `20261014_v3infer_v2ckpt_hazelnut_hole_100img` (checkpoint v2 + inference v3, để tách hiệu quả của phần inference).

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
