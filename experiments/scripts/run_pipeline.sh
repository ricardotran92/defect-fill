#!/usr/bin/env bash
# Train -> generate -> evaluate one committed code version of defectfill_02 (server).
#
#   VERSION=v3 bash experiments/scripts/run_pipeline.sh
#
# Optional: NIMG (default 100), GPU (2), OBJECT (hazelnut), DEFECT (hole),
#           TRAIN_ARGS / INFER_ARGS (extra flags passed to train.py / inference.py).
# Outputs: runs/<object>_<defect>_paper_2000step_<VERSION>, runs/<object>_<defect>_paper_generated_<VERSION>_<NIMG>,
#          a row "defectfill_02_<VERSION>_<NIMG>img" in runs/paper_protocol_metrics.csv.
set -eo pipefail
: "${VERSION:?set VERSION, e.g. VERSION=v3}"
NIMG=${NIMG:-100}; GPU=${GPU:-2}; OBJECT=${OBJECT:-hazelnut}; DEFECT=${DEFECT:-hole}
ROOT=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
cd "$ROOT"
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate ~/envs/defectfill
SHARED=$HOME/shared; TAG=defectfill_02_${VERSION}
RUN=runs/${OBJECT}_${DEFECT}_paper_2000step_${VERSION}
GEN=runs/${OBJECT}_${DEFECT}_paper_generated_${VERSION}_${NIMG}

if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "Uncommitted code changes: commit (and tag) before running, so the result is traceable." >&2
  exit 1
fi
echo "== $(date) pwd=$(pwd) branch=$(git branch --show-current) commit=$(git rev-parse --short HEAD) version=$VERSION GPU=$GPU images=$NIMG"

python tools/perf_run.py --gpu "$GPU" --tag "$TAG" -- python train.py \
  --data_dir "$SHARED/datasets/mvtec_ad" --object_class "$OBJECT" --defect_type "$DEFECT" \
  --output_dir "$RUN" --base_model_path "$SHARED/models/sd2_inpainting" $TRAIN_ARGS
python tools/perf_run.py --gpu "$GPU" --tag "$TAG" -- python inference.py \
  --checkpoint "$RUN/checkpoints/checkpoint_final.pt" --output_dir "$GEN" \
  --base_model_path "$SHARED/models/sd2_inpainting" --data_dir "$SHARED/datasets/mvtec_ad" \
  --object_class "$OBJECT" --defect_type "$DEFECT" \
  --total_images "$NIMG" --num_samples 8 --steps 50 --guidance_scale 2.0 $INFER_ARGS
python tools/perf_run.py --gpu "$GPU" --tag "$TAG" -- python evaluate.py \
  --generated_dir "$GEN" --split_manifest "$RUN/split_manifest.json" \
  --output_csv runs/paper_protocol_metrics.csv --object_class "$OBJECT" --defect_type "$DEFECT" \
  --config_name "${TAG}_${NIMG}img" --category_type object
echo "== DONE $VERSION $(date)"
