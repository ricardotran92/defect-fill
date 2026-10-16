# Lịch sử phiên bản code (`defectfill_02`)

Mỗi phiên bản là một tag `defectfill_02-vN` trên branch `defectfill_02`. Baseline là branch `main` (thuật toán của tag `baseline-original` = `c825dff`). Kết quả từng run nằm trong `RESULTS.md` và `records/`. Đối chiếu công thức paper ↔ code: `formula/formula_mapping.md` (bản sao local, không có trên git) và hướng dẫn server, Phần F.

Paper (hazelnut, trung bình mọi loại defect, 1000 ảnh): KID 1.13, IC-LPIPS 0.31.

## v7 — tăng trọng số $\mathcal L_{def}$ cho phần mảnh của mask (tag `defectfill_02-v7`)

**Vấn đề:** v6 có lỗ đẹp nhưng không có vết nứt ở nét mảnh. Chẩn đoán `diagnostics/20261016_attention_thin_masks` cho thấy token `sks` của v6 đã dồn khoảng 55% attention vào nét mảnh. Như vậy model biết **đặt** defect ở đó, nhưng không biết **vẽ gì**.

**Nguyên nhân:** trong 6 ảnh train, phần mảnh chỉ chiếm **14.8%** số ô mask ở mức latent (held-out: 31.2%). $\mathcal L_{def}$ vì thế gần như chỉ dạy cách vẽ lỗ tròn.

**Thay đổi:**
- `model.thin_mask_part(mask)` = mask trừ đi morphological opening 7×7 của nó, cùng định nghĩa với `tools/mask_fill.py`.
- `train.py --thin_weight w`: trọng số của $\mathcal L_{def}$ là $M + (w-1)\cdot\mathrm{thin}$ ở mức latent. Vì trọng số nằm trong chuẩn bình phương, sai số ở ô mảnh có trọng số $w^2$. Kênh mask đưa vào UNet, $\mathcal L_{obj}$ và $\mathcal L_{attn}$ không đổi.
- Mặc định `1.0` (tắt). v7 chạy với **`2.5`**: $\frac{2.5^2 \cdot 0.148}{2.5^2 \cdot 0.148 + 0.852} \approx 52\%$ $\mathcal L_{def}$ đến từ phần mảnh. Giá trị này có trong fingerprint resume.
- Ngoài ra giữ như v6. Tool chẩn đoán `tools/attention_probe.py` và `DefectFillModel.token_attention_map` có từ commit `9872fa8`.

**Ghi chú:** paper không có trọng số riêng cho phần mảnh. Đây là cách bù cho việc tập train ít ví dụ vết nứt, không phải tái hiện paper.

**Kết quả** (16/10/2026, 17:30 → 19:15, run `20261016_v7_hazelnut_hole_100img_priority`; hình `figures/20261016_thin_masks_baseline_v4_v5_v6_v7.jpg`):

| 100 ảnh, kế hoạch ưu tiên | v6 | **v7** |
|---|---|---|
| `fill_thin` | 6.0% | 5.3% |
| `fill_thick` | 40.9% | 36.3% |
| KID×1000 (±) | −1.77 (4.53) | 1.45 (5.64) |
| IC-LPIPS train / held-out | 0.252 / 0.212 | 0.248 / 0.211 |
| $\mathcal L_{def}$ / $\mathcal L_{attn}$ cuối | 0.117 / 0.019 | 0.123 / 0.020 |

- **Không cải thiện.** Lỗ sáng hơn v6 (lòng màu be thay vì tối sâu), và vết nứt rõ vẫn không có. Chỉ một mẫu có vệt đỏ nhạt dọc nhánh, nhưng mẫu đó lại mất lỗ.
- Tăng trọng số cho số ít pixel mảnh trong 6 ảnh train không dạy được model vẽ vết nứt: các pixel đó chủ yếu là mép lỗ và vài vệt ngắn, không phải vết nứt dài. Bước tiếp theo kiểm tra trực tiếp giả thuyết dữ liệu bằng một split có ảnh train chứa vết nứt (`diagnostics/20261016_split_thin_share`, run `v6split113`).

## v6 — chuẩn hóa $\mathcal L_{attn}$ theo vùng trong / ngoài mask (tag `defectfill_02-v6`)

**Vấn đề** (run `20261016_*_priority`): v5 cho lỗ tối, sắc, nhưng không có vết nứt ở nét mảnh và lỗ nhỏ hơn mask. $\mathcal L_{attn}$ được tính trung bình trên toàn bản đồ 64×64, nên độ lớn chỉ cỡ diện tích mask (khoảng 0.013). Khi $\mathcal L_{def}$ mạnh, $\mathcal L_{attn}$ gần như không giảm (v5: 0.018 → 0.013; baseline: 0.0134 → 0.0133). Chỉ ở v4, nơi $\mathcal L_{def}$ yếu, $\mathcal L_{attn}$ mới học được (0.009 → 0.003) và defect mới theo nét mảnh.

**Thay đổi:** `model.attention_mask_loss(A, M, "region")` = trung bình $(A-1)^2$ trong mask + trung bình $A^2$ ngoài mask. Độ lớn không còn phụ thuộc diện tích defect (khoảng 1 khi `sks` không chú ý vào mask, 0 khi khớp hoàn toàn).
- `train.py --attention_loss_norm region|total`, mặc định `region`, có trong fingerprint resume. `total` là cách của v1–v5.
- Mọi thứ khác giữ như v5 (`--masked_loss_norm mask`, mask latent max-pool, ảnh nền theo mask gốc, `GradScaler`).

**Kỳ vọng:** lỗ tối, sắc như v5 (nhờ $\mathcal L_{def}$), đồng thời phủ đúng hình mask và có vết nứt ở nét mảnh (nhờ $\mathcal L_{attn}$ học được). Paper Hình 7 cũng nói bỏ $\mathcal L_{attn}$ thì "incomplete hole".

**Rủi ro:** $\lambda_{attn}=0.05$ nhân với $\mathcal L_{attn}$ khoảng 1 cho đóng góp ngang $\lambda_{def}\mathcal L_{def}$ (khoảng 0.06). Nếu attention quá mạnh, ảnh có thể bị vỡ trong mask; theo dõi bằng ảnh và `fill_*`.

**Kết quả** (16/10/2026, run `20261016_v6_hazelnut_hole_100img_priority`, cùng kế hoạch 100 ảnh ưu tiên 3:1 và cùng cặp ảnh/mask/seed với ba bộ của v5; hình: `figures/20261016_thin_masks_baseline_v4_v5_v6.jpg`):

| 100 ảnh, kế hoạch ưu tiên | baseline | v4 | v5 | **v6** |
|---|---|---|---|---|
| `fill_thin` | 5.8% | 21.0% | 3.3% | 6.0% |
| `fill_thick` | 45.4% | 46.5% | 33.3% | 40.9% |
| `drift_far` | 2.17 | 0 | 0 | 0 |
| KID×1000 (±) | −1.20 (4.23) | 2.30 (5.30) | 0.94 (5.47) | −1.77 (4.53) |
| IC-LPIPS, tâm = ảnh train | 0.249 | 0.249 | 0.247 | 0.252 |
| $\mathcal L_{attn}$ khi train (đầu → cuối, mỗi 400 bước) | 0.0134 → 0.0133 | 0.0091 → 0.0029 | 0.0181 → 0.0132 | **0.240 → 0.019** (thang đo theo vùng) |
| $\mathcal L_{def}$ khi train | 1.16 → 0.52 | 0.0049 → 0.0023 | 0.263 → 0.117 | 0.267 → 0.117 |

- $\mathcal L_{attn}$ học rất mạnh, và $\mathcal L_{def}$ vẫn mạnh như v5. Train 1:17:20 (GPU 2 không có ai dùng chung), `LossScale` không giảm lần nào.
- **Lỗ:** to hơn v5, phủ phần thân mask tốt hơn. Mép không đều, có viền gỗ sáng sần sùi, bên trong tối có chiều sâu. Đây là bộ giống ảnh thật và Hình 6 nhất từ trước tới nay.
- **Vết nứt ở nét mảnh vẫn không có** (`fill_thin` 6%). Attention đã được học theo đúng mask mà model vẫn không vẽ ở nét mảnh, nên nguyên nhân không nằm ở $\mathcal L_{attn}$.
- **Nguyên nhân khả dĩ nhất là dữ liệu:** 6 ảnh train của hazelnut/hole gần như không có nét mảnh (2.8% diện tích mask, so với 10.3% ở held-out). Model học "hole" là lỗ tròn, và không có ví dụ nào cho "vết nứt mảnh".

**Hướng tiếp theo (chưa chạy):**
1. Xem trực tiếp bản đồ attention của `sks` ở nét mảnh trên ảnh sinh (chạy trên server, rẻ). Mục đích là xác nhận attention đã phủ nét mảnh, và nhánh vẽ mới là chỗ thiếu.
2. Tăng trọng số $\mathcal L_{def}$ cho phần mảnh của mask (tách bằng morphological opening, như `tools/mask_fill.py`).
3. Kiểm tra split: `split_seed` hiện tại cho 6 ảnh train ít nét mảnh. Với split khác, paper có thể đã có ảnh train chứa vết nứt. Đổi split làm thay đổi protocol, nên chỉ dùng để chẩn đoán.

## v5 — chuẩn hóa loss có mask theo trọng số trong mask (tag `defectfill_02-v5`)

**Vấn đề:** từ v2 đến v4, defect sinh ra nhạt và mờ, nét mảnh chỉ là vệt sáng. Paper (Hình 6, Hình S9) cho lỗ đen sâu, mép sắc, và nét mảnh thành vết nứt tối. Ảnh train thật cũng như vậy: lỗ tối, viền gỗ sáng sần sùi, vết nứt mảnh tối.

**Loại trừ guidance scale** (quét trên PC ngày 16/10, checkpoint v4, 4 cặp, w = 2 / 5 / 7.5 / 10): guidance cao **không** làm defect rõ hơn. Nét mảnh vỡ thành mảnh màu tím xanh, và bên trong lỗ ra màu lạ. w = 2 là sạch nhất, nên giữ w = 2.

**Nguyên nhân (giả thuyết, cần run để xác nhận):** cách chuẩn hóa L1 của v1 (chia cho số phần tử của toàn tensor) làm $\mathcal L_{def}$ tỉ lệ với diện tích defect. Mask train của hazelnut/hole chỉ chiếm 1.2% diện tích ảnh, nên $\mathcal L_{def}$ chỉ còn khoảng 1% độ lớn.

| | Trọng số 1 pixel defect : 1 pixel nền |
|---|---|
| Baseline (chia theo diện tích mask, có lỗi ×4) | khoảng 460 : 1 |
| v2–v4 (chia cho toàn tensor) | khoảng 39 : 1 |

Nhánh object (tái tạo nền quanh box ngẫu nhiên) lấn át, nên model "tái tạo nhiều hơn tạo defect". Đây đúng là hiện tượng paper mô tả ở Hình 7 khi bỏ $\mathcal L_{def}$. Mask train còn ít nét mảnh: phần mảnh chiếm 2.8% mask train, so với 10.3% ở mask held-out.

**Thay đổi:**

- `compute_masked_mse(..., normalize)`. Với `"mask"` (mặc định), loss là $\sum (W\odot r)^2 / \sum W^2$ trên mọi kênh: trung bình có trọng số trong mask. $M'$ vẫn nằm trong chuẩn bình phương (nền có trọng số α²), và không còn lỗi ×4 của baseline.
- `train.py --masked_loss_norm mask|total`, có trong fingerprint resume. `total` là cách của v2–v4.
- Paper không nói cách chuẩn hóa. Đây là lựa chọn theo bằng chứng thực nghiệm; nó thay thế cách đọc "sát chữ" đã chọn ở v1.

**Kết quả** (16/10/2026, kế hoạch 100 ảnh ưu tiên mask nét mảnh 3:1 của v5.1 `71e1332`; ba bộ dùng cùng cặp ảnh/mask/seed; hình: `figures/20261016_thin_masks_baseline_v4_v5.jpg`):

| 100 ảnh, kế hoạch ưu tiên | baseline (checkpoint gốc) | v4 | v5 |
|---|---|---|---|
| Nét mảnh được vẽ (`fill_thin`) | 5.8% | **21.0%** | 3.3% |
| Phần dày được vẽ (`fill_thick`) | 45.4% | 46.5% | 33.3% |
| Nền xa mask đổi (`drift_far`) | 2.17 | 0 | 0 |
| KID×1000 (±) | −1.20 (4.23) | 2.30 (5.30) | 0.94 (5.47) |
| IC-LPIPS, tâm = ảnh train | 0.249 | 0.249 | 0.247 |
| $\mathcal L_{def}$ khi train (đầu → cuối) | 1.16 → 0.52 | 0.0049 → 0.0023 | 0.263 → 0.117 |
| $\mathcal L_{attn}$ khi train (đầu → cuối) | 0.0134 → 0.0133 | **0.0091 → 0.0029** | 0.0181 → 0.0132 |

- **Giả thuyết đúng một nửa.** Khi $\mathcal L_{def}$ đủ mạnh, lỗ **tối, sắc, tròn**, giống ảnh thật và Hình 6 hơn v4. Nhưng nét mảnh **không** được vẽ, và lỗ thường nhỏ hơn mask.
- **Nguyên nhân phần còn thiếu:** $\mathcal L_{attn}$ mới là thứ kéo defect theo đúng hình mask (paper Hình 7: bỏ $\mathcal L_{attn}$ thì "incomplete hole"). $\mathcal L_{attn}$ được tính trung bình trên **toàn ảnh**, nên độ lớn chỉ cỡ diện tích mask (khoảng 0.013). Ở baseline và v5 nó bị lấn át và gần như không giảm. Ở v4 nó học mạnh, nên mới có vệt theo nét mảnh, nhưng khi đó $\mathcal L_{def}$ quá yếu nên defect nhạt.
- KID và IC-LPIPS vẫn trong vùng nhiễu, không phân biệt được ba bộ.

**Hướng tiếp theo (v6, chưa chạy):** chuẩn hóa $\mathcal L_{attn}$ theo vùng, cùng logic với v5. Lấy trung bình $(A-1)^2$ **trong mask** cộng trung bình $A^2$ **ngoài mask**, thay cho trung bình trên toàn ảnh. Khi đó $\mathcal L_{attn}$ có độ lớn ngang $\mathcal L_{def}$, nên defect vừa tối/sắc (nhờ $\mathcal L_{def}$) vừa phủ đúng hình mask và nét mảnh (nhờ $\mathcal L_{attn}$).

Run: `20261016_v5_hazelnut_hole_100img_priority`, cùng hai bộ so sánh `20261016_v4ckpt_…_priority` và `20261016_baselineckpt_…_priority`.

### v5.1 (`71e1332`, tag `defectfill_02-v5.1`), không đổi thuật toán

- `inference.py --priority_masks` / `--priority_ratio`: kế hoạch sinh ảnh ưu tiên mask chỉ định.
- Script server có thêm `GEN_SUFFIX` và `CKPT_RUN`.

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
