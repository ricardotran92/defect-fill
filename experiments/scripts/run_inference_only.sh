#!/usr/bin/env bash
# Generate -> evaluate with the current code, reusing the checkpoint of an earlier version (server).
# Isolates inference-side changes from training-side changes.
#
#   CKPT_VERSION=v2 NAME=v3infer_v2ckpt bash experiments/scripts/run_inference_only.sh
#
# Optional: NIMG (100), GPU (2), OBJECT (hazelnut), DEFECT (hole), INFER_ARGS,
#           GEN_SUFFIX (appended to the output dir and config name),
#           CKPT_RUN (training run dir holding checkpoints/checkpoint_final.pt; default: this worktree's
#           runs/<object>_<defect>_paper_2000step_<CKPT_VERSION>, e.g. the baseline run in ~/projects/defect-fill).
# Outputs: runs/<object>_<defect>_paper_generated_<NAME>_<NIMG>, metrics row "defectfill_02_<NAME>_<NIMG>img".
set -eo pipefail
: "${CKPT_VERSION:?set CKPT_VERSION, e.g. v2}"
: "${NAME:?set NAME, e.g. v3infer_v2ckpt}"
NIMG=${NIMG:-100}; GPU=${GPU:-2}; OBJECT=${OBJECT:-hazelnut}; DEFECT=${DEFECT:-hole}
ROOT=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
cd "$ROOT"
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate ~/envs/defectfill
SHARED=$HOME/shared; TAG=defectfill_02_${NAME}
RUN=${CKPT_RUN:-runs/${OBJECT}_${DEFECT}_paper_2000step_${CKPT_VERSION}}
GEN=runs/${OBJECT}_${DEFECT}_paper_generated_${NAME}_${NIMG}${GEN_SUFFIX}

if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "Uncommitted code changes: commit (and tag) before running, so the result is traceable." >&2
  exit 1
fi
[ -f "$RUN/checkpoints/checkpoint_final.pt" ] || { echo "Missing $RUN/checkpoints/checkpoint_final.pt" >&2; exit 1; }
echo "== $(date) pwd=$(pwd) branch=$(git branch --show-current) commit=$(git rev-parse --short HEAD) name=$NAME checkpoint=$RUN GPU=$GPU images=$NIMG"

python tools/perf_run.py --gpu "$GPU" --tag "$TAG" -- python inference.py \
  --checkpoint "$RUN/checkpoints/checkpoint_final.pt" --output_dir "$GEN" \
  --base_model_path "$SHARED/models/sd2_inpainting" --data_dir "$SHARED/datasets/mvtec_ad" \
  --object_class "$OBJECT" --defect_type "$DEFECT" \
  --total_images "$NIMG" --num_samples 8 --steps 50 --guidance_scale 2.0 $INFER_ARGS
python tools/perf_run.py --gpu "$GPU" --tag "$TAG" -- python evaluate.py \
  --generated_dir "$GEN" --split_manifest "$RUN/split_manifest.json" \
  --output_csv runs/paper_protocol_metrics.csv --object_class "$OBJECT" --defect_type "$DEFECT" \
  --config_name "${TAG}_${NIMG}img${GEN_SUFFIX}" --category_type object
echo "== DONE $NAME $(date)"
