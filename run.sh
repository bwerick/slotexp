#!/usr/bin/env bash
# =============================================================================
#  slot_occlusion: single-command pipeline
#  Usage: bash run.sh [--epochs N] [--fast] [--gif]
#
#  Does everything:
#    1. Creates conda environment + installs deps
#    2. Generates synthetic occlusion dataset
#    3. Trains vanilla (baseline) Slot Attention model
#    4. Trains temporal (proposed) Slot Attention model
#    5. Generates all paper figures
#
#  Requirements: conda, CUDA GPU recommended (works on CPU but slow)
# =============================================================================

set -e  # exit on error

# ── Parse args ────────────────────────────────────────────────────────────────
EPOCHS=100
FAST=0
GIF_FLAG=""

for arg in "$@"; do
  case $arg in
    --epochs=*) EPOCHS="${arg#*=}" ;;
    --fast)     FAST=1 ;;          # quick smoke-test: small data, few epochs
    --gif)      GIF_FLAG="--gif" ;;
  esac
done

if [ "$FAST" = "1" ]; then
  EPOCHS=10
  N_VIDEOS=200
  BATCH_SIZE=16
  echo "⚡ FAST mode: epochs=$EPOCHS, videos=$N_VIDEOS"
else
  N_VIDEOS=2000
  BATCH_SIZE=32
fi

ENV_NAME="slots"

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║   Slot Attention Occlusion Experiment Pipeline       ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
echo "Config:"
echo "  Epochs:    $EPOCHS"
echo "  Videos:    $N_VIDEOS (+ 200 val)"
echo "  Batch:     $BATCH_SIZE"
echo ""

# ── Step 1: Environment ───────────────────────────────────────────────────────
echo "━━━ Step 1/5: Environment setup ━━━━━━━━━━━━━━━━━━━━━━━"

if conda env list | grep -q "^${ENV_NAME} "; then
  echo "✓ Conda env '$ENV_NAME' already exists"
else
  echo "Creating conda env: $ENV_NAME (Python 3.10)..."
  conda create -n "$ENV_NAME" python=3.10 -y
fi

# Activate
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

echo "Installing dependencies..."
pip install -q --upgrade pip

# Detect CUDA version for correct torch install
CUDA_VER=$(python -c "import subprocess; r=subprocess.run(['nvcc','--version'],capture_output=True,text=True); print('cu121' if '12.' in r.stdout else 'cu118' if '11.' in r.stdout else 'cpu')" 2>/dev/null || echo "cpu")

if [ "$CUDA_VER" = "cpu" ]; then
  echo "⚠  No CUDA detected — installing CPU PyTorch (training will be slow)"
  pip install -q torch torchvision
else
  echo "✓ CUDA detected ($CUDA_VER) — installing GPU PyTorch"
  pip install -q torch torchvision --index-url "https://download.pytorch.org/whl/${CUDA_VER}"
fi

pip install -q numpy matplotlib imageio

echo "✓ Environment ready"
echo ""

# ── Step 2: Generate dataset ──────────────────────────────────────────────────
echo "━━━ Step 2/5: Generating synthetic occlusion dataset ━━━"

if [ -f "data/train/frames.npy" ] && [ -f "data/val/frames.npy" ]; then
  echo "✓ Dataset already exists (delete data/ to regenerate)"
else
  python generate_data.py \
    --out_dir data/train \
    --n_videos "$N_VIDEOS" \
    --canvas_size 64 \
    --n_frames 30 \
    --n_objects 3 \
    --radius 8 \
    --seed 42
fi
echo ""

# ── Step 3: Train vanilla model ───────────────────────────────────────────────
echo "━━━ Step 3/5: Training vanilla (baseline) model ━━━━━━━━"

if [ -f "checkpoints/vanilla_best.pt" ]; then
  echo "✓ Vanilla checkpoint exists (delete checkpoints/vanilla_best.pt to retrain)"
else
  python train.py \
    --model vanilla \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr 4e-4 \
    --n_slots 4 \
    --slot_dim 64 \
    --encoder_hidden 64 \
    --n_iters 3 \
    --n_frames 30 \
    --canvas_size 64
fi
echo ""

# ── Step 4: Train temporal model ──────────────────────────────────────────────
echo "━━━ Step 4/5: Training temporal (proposed) model ━━━━━━━"

if [ -f "checkpoints/temporal_best.pt" ]; then
  echo "✓ Temporal checkpoint exists (delete checkpoints/temporal_best.pt to retrain)"
else
  python train.py \
    --model temporal \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr 4e-4 \
    --n_slots 4 \
    --slot_dim 64 \
    --encoder_hidden 64 \
    --n_iters 3 \
    --n_frames 30 \
    --canvas_size 64
fi
echo ""

# ── Step 5: Visualize ─────────────────────────────────────────────────────────
echo "━━━ Step 5/5: Generating visualizations ━━━━━━━━━━━━━━━━"

python visualize.py \
  --seed 1337 \
  --n_objects 3 \
  --canvas_size 64 \
  --n_frames 30 \
  $GIF_FLAG

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║   ✅  Pipeline complete!                             ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
echo "Paper figures saved to outputs/:"
echo "  occlusion_comparison.png  — main result figure"
echo "  slot_drift_analysis.png   — entropy/binding drift"
echo "  reconstruction_grid.png   — per-slot breakdown"
if [ -n "$GIF_FLAG" ]; then
echo "  occlusion_summary.gif     — animated walkthrough"
fi
echo ""
echo "To explore more test cases:"
echo "  conda activate $ENV_NAME"
echo "  python visualize.py --seed 42"
echo "  python visualize.py --seed 999"
