#!/bin/bash
# ============================================================================
# RunPod Setup Script — RTX A4500 / 12 vCPU / 62 GB RAM
# Run this ONCE after deploying the pod and cloning the repo.
# ============================================================================
set -e

echo "=== Tabula Datagen Pod Setup ==="

# 1. Install system deps
apt-get update -qq && apt-get install -y -qq git curl htop tmux > /dev/null

# 2. Clone repo if not present (uses GH_PAT for avewright account)
REPO_DIR="/workspace/tabula"
if [ ! -d "$REPO_DIR" ]; then
    echo "Cloning tabula repo from avewright (not avewright-kahua)..."
    if [ -n "$GH_PAT" ]; then
        git clone "https://${GH_PAT}@github.com/avewright/tabula.git" "$REPO_DIR"
    else
        echo "ERROR: GH_PAT not set. Export it first:"
        echo "  export GH_PAT=ghp_YOUR_TOKEN"
        exit 1
    fi
fi
cd "$REPO_DIR"

# Set git remote to use PAT for avewright account
if [ -n "$GH_PAT" ]; then
    git remote set-url origin "https://${GH_PAT}@github.com/avewright/tabula.git"
fi

# 3. Create venv and install
python -m venv /workspace/venv --system-site-packages
source /workspace/venv/bin/activate

pip install --quiet --upgrade pip setuptools wheel
pip install --quiet -e ".[dev]"

# 4. Install Tier-2 neural generators (A4500 has 20GB VRAM)
pip install --quiet ctgan sdv tab-ddpm 2>/dev/null || echo "WARN: Some Tier-2 deps failed (non-fatal)"
pip install --quiet transformers accelerate 2>/dev/null || echo "WARN: transformers install issue (non-fatal)"

# 5. Create .env if missing
if [ ! -f "$REPO_DIR/.env" ]; then
    echo "Creating .env template — fill in your tokens!"
    cat > "$REPO_DIR/.env" <<EOF
HF_TOKEN=hf_YOUR_TOKEN_HERE
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_key
GH_PAT=${GH_PAT:-ghp_YOUR_PAT_HERE}
EOF
    echo "IMPORTANT: Edit /workspace/tabula/.env with real credentials before running."
fi

# 6. Verify GPU
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_mem/1e9:.1f} GB')" 2>/dev/null || echo "WARN: No GPU detected"

echo ""
echo "=== Setup complete ==="
echo "Git remote set to: avewright/tabula (via GH_PAT)"
echo "HF push target: avewright/tabula-pretraining-corpus"
echo ""
echo "To run:"
echo "  source /workspace/venv/bin/activate && cd /workspace/tabula && python run_datagen_pod.py"
echo "Use tmux for persistence:"
echo "  tmux new -s datagen"
