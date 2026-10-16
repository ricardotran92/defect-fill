# Chẩn đoán (không phải run sinh ảnh)

Mỗi thư mục con là một lần chẩn đoán, gồm output nhỏ (JSON, ảnh JPEG) và `commit.txt` (commit code đã chạy). Lệnh chạy và kết luận ghi ở bảng dưới. Mọi chẩn đoán chạy trên server, GPU 2.

| Thư mục | Ngày | Công cụ | Câu hỏi | Kết luận |
|---|---|---|---|---|
| [`20261016_attention_thin_masks`](20261016_attention_thin_masks/) | 16/10/2026 | `tools/attention_probe.py` (commit `9872fa8`) | Token `sks` có chú ý vào nét mảnh của mask không (checkpoint baseline, v4, v5, v6; 5 mask mảnh × 2 ảnh normal × t = 200/500/800)? | **Có, với v6.** Attention trung bình ở phần mảnh / phần dày / nền xa: baseline 0.034 / 0.040 / 0.014; v4 0.151 / 0.175 / 0.006; v5 0.066 / 0.143 / 0.012; **v6 0.547 / 0.732 / 0.047**. v6 dồn khoảng 55% attention vào nét mảnh, và heatmap bám theo cả các nhánh. Vậy việc v6 không vẽ vết nứt **không phải do attention**, mà do model không biết vẽ gì ở nét mảnh: vết nứt mảnh gần như không có trong 6 ảnh train. Hướng xử lý là v7 (tăng trọng số $\mathcal L_{def}$ cho phần mảnh). |

Lệnh của lần chẩn đoán đầu tiên (thư mục `~/projects/defectfill_02`):

```bash
B=$HOME/projects/defect-fill/runs/hazelnut_hole_paper_2000step/checkpoints/checkpoint_final.pt; R=runs
CUDA_VISIBLE_DEVICES=2 python tools/attention_probe.py --checkpoint baseline=$B \
  --checkpoint v4=$R/hazelnut_hole_paper_2000step_v4/checkpoints/checkpoint_final.pt \
  --checkpoint v5=$R/hazelnut_hole_paper_2000step_v5/checkpoints/checkpoint_final.pt \
  --checkpoint v6=$R/hazelnut_hole_paper_2000step_v6/checkpoints/checkpoint_final.pt \
  --base_model_path $HOME/shared/models/sd2_inpainting --data_dir $HOME/shared/datasets/mvtec_ad \
  --split_manifest $R/hazelnut_hole_paper_2000step_v6/split_manifest.json \
  --mask_names 006_mask.png,010_mask.png,011_mask.png,014_mask.png,016_mask.png --good_indices 0,1 \
  --timesteps 200,500,800 --output_dir runs/diagnostics/20261016_attention_thin_masks
```

`attention_grid.jpg`: mỗi hàng là một cặp (ảnh normal, mask); các cột là ảnh kèm viền mask, rồi heatmap của baseline, v4, v5, v6. Mỗi heatmap được chia cho giá trị lớn nhất của chính nó.
