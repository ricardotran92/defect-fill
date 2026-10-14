#!/usr/bin/env bash
# Copy the light artifacts of one server run to the matching PC worktree, for tools/record_run.py (PC, Git Bash).
#
#   bash experiments/scripts/fetch_run.sh defectfill_02 hazelnut_hole_paper_2000step_v3 hazelnut_hole_paper_generated_v3_100
#
# Args: WORKTREE (defectfill_02 | defect-fill), TRAIN_DIR (run dir name, or - for inference-only runs), GEN_DIR.
# Copies: train_config.json, split_manifest.json, tensorboard/, train_log.txt, the generated folder (~61 MB per
# 100 images), paper_protocol_metrics.csv, compare/*.csv and perf/perf_summary.csv. Checkpoints are not copied.
set -eo pipefail
WORKTREE=${1:?worktree name}; TRAIN_DIR=${2:?train run dir or -}; GEN_DIR=${3:?generated dir}
SERVER=${SERVER:-ricardo@140.123.104.126}
REMOTE=projects/$WORKTREE/runs
LOCAL=${LOCAL_ROOT:-/d/repos}/$WORKTREE/runs
mkdir -p "$LOCAL/compare" "$LOCAL/perf"

if [ "$TRAIN_DIR" != "-" ]; then
  mkdir -p "$LOCAL/$TRAIN_DIR"
  scp -q -r "$SERVER:$REMOTE/$TRAIN_DIR/train_config.json" "$SERVER:$REMOTE/$TRAIN_DIR/split_manifest.json" \
    "$SERVER:$REMOTE/$TRAIN_DIR/tensorboard" "$SERVER:$REMOTE/$TRAIN_DIR/train_log.txt" "$LOCAL/$TRAIN_DIR/"
fi
scp -q -r "$SERVER:$REMOTE/$GEN_DIR" "$LOCAL/"
scp -q "$SERVER:$REMOTE/paper_protocol_metrics.csv" "$LOCAL/paper_protocol_metrics_server.csv"
scp -q "$SERVER:$REMOTE/perf/perf_summary.csv" "$LOCAL/perf/perf_summary_server.csv"
scp -q "$SERVER:$REMOTE/compare/*.csv" "$LOCAL/compare/" 2>/dev/null || true
echo "fetched $WORKTREE: ${TRAIN_DIR} ${GEN_DIR} -> $LOCAL"
