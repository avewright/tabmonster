#!/usr/bin/env bash
# Quick status dashboard for both tmux processes
# Usage: bash status.sh  (or: watch -n 10 bash status.sh)

set -euo pipefail
cd "$(dirname "$0")"

SEP="════════════════════════════════════════════════════════════════════════"

echo "$SEP"
echo "  TABMONSTER STATUS  $(date '+%Y-%m-%d %H:%M:%S')"
echo "$SEP"

# ── Pretraining ──────────────────────────────────────────────
echo ""
echo "▶ PRETRAINING (tmux: pretrain)"
echo "──────────────────────────────────────────────────────────"
if tmux has-session -t pretrain 2>/dev/null; then
    LAST=$(tail -1 artifacts/pretrain_corpus_v1_output.log 2>/dev/null || echo "no log")
    echo "  Session: ALIVE"
    echo "  Latest:  $LAST"
    # Extract key metrics
    STEP=$(echo "$LAST" | grep -oP 'step=\s*\K[\d,]+' || echo "?")
    LOSS=$(echo "$LAST" | grep -oP 'loss=\K[\d.]+' || echo "?")
    ROWS=$(echo "$LAST" | grep -oP 'rows=\K[\d,]+' || echo "?")
    RATE=$(echo "$LAST" | grep -oP 'rate=\K[\d,]+' || echo "?")
    ETA=$(echo "$LAST" | grep -oP 'eta=\K[\d.]+h' || echo "?")
    echo "  Step:    $STEP / 200,000"
    echo "  Loss:    $LOSS"
    echo "  Rows:    $ROWS"
    echo "  Rate:    $RATE rows/s"
    echo "  ETA:     $ETA"
    # Best val
    BEST=$(grep -a "best=" artifacts/pretrain_corpus_v1_output.log 2>/dev/null | tail -1 || echo "")
    if [[ -n "$BEST" ]]; then
        echo "  Val:     $BEST"
    fi
    # Checkpoint info
    if [[ -f artifacts/pretrain_corpus_v1/latest.pt ]]; then
        CKPT_TIME=$(stat -c '%y' artifacts/pretrain_corpus_v1/latest.pt 2>/dev/null | cut -d. -f1)
        CKPT_SIZE=$(stat -c '%s' artifacts/pretrain_corpus_v1/latest.pt 2>/dev/null)
        echo "  Ckpt:    latest.pt ($(numfmt --to=iec "$CKPT_SIZE"), saved $CKPT_TIME)"
    fi
else
    echo "  Session: DOWN"
fi

# ── Data Loop ────────────────────────────────────────────────
echo ""
echo "▶ DATA LOOP (tmux: dataloop)"
echo "──────────────────────────────────────────────────────────"
if tmux has-session -t dataloop 2>/dev/null; then
    echo "  Session: ALIVE"
    # Latest shard
    SHARD_LINE=$(grep -a ">>> SHARD" corpus/real_data/loop_output.log 2>/dev/null | tail -1 || echo "none")
    echo "  Latest:  $SHARD_LINE"
    # Round info
    ROUND_LINE=$(grep -a "^ROUND" corpus/real_data/loop_output.log 2>/dev/null | tail -1 || echo "none")
    echo "  Round:   $ROUND_LINE"
    # Total rows from round line
    TOTAL_ROWS=$(echo "$ROUND_LINE" | grep -oP 'rows=\K[\d,]+' || echo "?")
    TOTAL_SHARDS=$(echo "$ROUND_LINE" | grep -oP 'shards=\K[\d,]+' || echo "?")
    ELAPSED=$(echo "$ROUND_LINE" | grep -oP 'elapsed=\K\d+min' || echo "?")
    echo "  Rows:    $TOTAL_ROWS"
    echo "  Shards:  $TOTAL_SHARDS (this session)"
    echo "  Elapsed: $ELAPSED"
    # Disk
    DISK=$(echo "$ROUND_LINE" | grep -oP 'disk=\K[\d.]+GB' || echo "?")
    echo "  Disk:    $DISK"
else
    echo "  Session: DOWN"
fi

# ── GPU ──────────────────────────────────────────────────────
echo ""
echo "▶ GPU"
echo "──────────────────────────────────────────────────────────"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu \
    --format=csv,noheader,nounits 2>/dev/null | while IFS=, read -r name mem_used mem_total util temp; do
    echo "  $name  ${mem_used}MB / ${mem_total}MB  util=${util}%  temp=${temp}°C"
done || echo "  nvidia-smi not available"

echo ""
echo "$SEP"
