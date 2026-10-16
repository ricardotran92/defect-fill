# Tỉ lệ nét mảnh theo mask và theo split (hazelnut/hole)

Tính trên CPU của server ngày 16/10/2026. Nét mảnh = mask trừ morphological opening 7×7, ảnh resize về 512 px (cùng định nghĩa với `tools/mask_fill.py`). `protocol.build_split_manifest` chia: shuffle với `random.Random(seed)`, 1/3 đầu là train.

**Phần mảnh trong từng mask (% diện tích mask):**
006 31.1 · 011 25.7 · 010 21.9 · 014 15.7 · 016 14.0 · 012 9.6 · 002 8.4 · 009 3.4 · 013 2.8 · 008 1.8 · 001 1.8 · 017 1.1 · 003 0.9 · 005 0.8 · 000 0.7 · 015 0.5 · 004 0.4 · 007 0.2

**Một số split** (trung bình phần mảnh của 6 mask train / 12 mask held-out):

| split_seed | train | held-out | Mask ưu tiên còn ở held-out | Mask train |
|---|---|---|---|---|
| **42 (mặc định, mọi run đến v7)** | **2.8%** | 10.3% | 5/5 | 013, 012, 004, 009, 005, 007 |
| 166 | 16.4% | 3.5% | 2/5 | 006, 002, 008, 011, 012, 010 |
| 109 | 16.3% | 3.6% | 2/5 | 005, 002, 010, 011, 012, 006 |
| 180 | 16.1% | 3.7% | 1/5 | 006, 016, 010, 013, 005, 011 |
| **113 (bước 3)** | 14.1% | 4.7% | 2/5 (**006**, 016) | 009, 012, 010, 011, 002, 014 |

**Kết luận:** seed 42 đặt cả 5 mask nhiều nét mảnh nhất vào held-out, nên 6 ảnh train gần như không có vết nứt. Đây là một trong những split bất lợi nhất cho việc học vết nứt. Paper chia ngẫu nhiên và không nêu seed.

**Bước 3** chạy `--split_seed 113` với cấu hình loss của v6 (`--thin_weight 1.0`), sinh ảnh ưu tiên 006 và 016, để xem model có vẽ vết nứt khi tập train có ví dụ vết nứt hay không. Vì đổi split, run này **chỉ dùng để chẩn đoán**, không dùng để báo kết quả theo protocol.
