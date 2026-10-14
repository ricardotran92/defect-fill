#!/usr/bin/env bash
# Score a defectfill_02 output with the BASELINE evaluate.py (server). The two evaluate.py versions
# define IC-LPIPS clusters differently (held-out vs training references), see the project guide E4.
#
#   GEN_NAME=v3 RUN_VERSION=v3 bash experiments/scripts/run_xeval.sh
#
# GEN_NAME: suffix of runs/<object>_<defect>_paper_generated_<GEN_NAME>_<NIMG> in this worktree.
# RUN_VERSION: training run whose split_manifest.json the output was generated with.
# Writes a row "<GEN_NAME>_out__baseline_eval_<NIMG>img" to ~/projects/defect-fill/runs/compare/xeval_<NIMG>.csv.
set -eo pipefail
: "${GEN_NAME:?set GEN_NAME, e.g. v3}"
: "${RUN_VERSION:?set RUN_VERSION, e.g. v3}"
NIMG=${NIMG:-100}; GPU=${GPU:-2}; OBJECT=${OBJECT:-hazelnut}; DEFECT=${DEFECT:-hole}
NEW=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
BASELINE=${BASELINE:-$HOME/projects/defect-fill}
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate ~/envs/defectfill
GEN=$NEW/runs/${OBJECT}_${DEFECT}_paper_generated_${GEN_NAME}_${NIMG}
RUN=$NEW/runs/${OBJECT}_${DEFECT}_paper_2000step_${RUN_VERSION}

cd "$BASELINE"
[ "$(git branch --show-current)" = main ] || { echo "$BASELINE is not the baseline (main) worktree" >&2; exit 1; }
mkdir -p runs/compare
echo "== $(date) xeval $GEN with baseline evaluate.py (commit $(git rev-parse --short HEAD))"
python tools/perf_run.py --gpu "$GPU" --tag "xeval_${GEN_NAME}-out" -- python evaluate.py \
  --generated_dir "$GEN" --split_manifest "$RUN/split_manifest.json" \
  --output_csv "runs/compare/xeval_${NIMG}.csv" --object_class "$OBJECT" --defect_type "$DEFECT" \
  --config_name "${GEN_NAME}_out__baseline_eval_${NIMG}img" --category_type object
echo "== DONE xeval $GEN_NAME $(date)"
