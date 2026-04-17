# Slot Attention Occlusion Experiment

Implements and compares two Slot Attention variants on synthetic occlusion scenes:

- **Vanilla**: frame-by-frame, no temporal memory (baseline — exhibits binding drift)
- **Temporal**: slots initialized from previous frame (temporal identity propagation, Eq. 8 from Chung et al.)

## Single Command

```bash
bash run.sh
```

That's it. It will:
1. Create a conda environment and install all dependencies
2. Generate a synthetic dataset of colored shapes with occlusion events
3. Train the vanilla (baseline) model
4. Train the temporal (proposed) model
5. Generate all paper figures in `outputs/`

## Options

```bash
bash run.sh --epochs=50        # fewer epochs (default: 100)
bash run.sh --fast             # smoke test: 10 epochs, 200 videos
bash run.sh --gif              # also generate animated GIF
```

## Expected Runtime (on 4090)

| Step | Time |
|------|------|
| Data generation (2000 videos) | ~2 min |
| Train vanilla (100 epochs) | ~20 min |
| Train temporal (100 epochs) | ~25 min |
| Visualization | ~1 min |
| **Total** | **~50 min** |

On a 3070: ~2 hours total.

## Output Figures

| File | Description |
|------|-------------|
| `outputs/occlusion_comparison.png` | Main result: slot masks vanilla vs temporal across occlusion frames |
| `outputs/slot_drift_analysis.png` | Attention entropy over time — shows binding drift during occlusion |
| `outputs/reconstruction_grid.png` | Per-slot RGB reconstructions across selected frames |
| `outputs/occlusion_summary.gif` | Animated walkthrough (requires `--gif` flag) |

## What This Demonstrates

The experiment directly demonstrates the core problem and partial solution described in the paper:

**Vanilla Slot Attention** (frame-by-frame):
- Slot bound to object 0 (red circle) shows spiking attention entropy during occlusion
- On reappearance, slot may fail to rebind correctly → binding drift

**Temporal Slot Attention** (with memory):
- Previous frame's slots used as initialization → strong prior toward owned object
- Entropy stays more stable during occlusion
- More reliable rebinding on reappearance

This implements Equation 8 from Chung et al. (Embodied-SlotSSM):

```
s_t^(0) = RandomInit()     if t == 0
s_t^(0) = s_{t-1}^(T)     if t > 0
```

## Re-running Visualization with Different Seeds

```bash
conda activate slot_occlusion
python visualize.py --seed 42
python visualize.py --seed 999
python visualize.py --seed 12345
```

## File Structure

```
slot_occlusion/
├── run.sh              ← single command entry point
├── generate_data.py    ← synthetic occlusion dataset generator
├── model.py            ← SlotAutoencoder + TemporalSlotAutoencoder
├── train.py            ← training loop for both models
├── visualize.py        ← generates all paper figures
├── data/
│   ├── train/          ← generated training data
│   └── val/            ← generated validation data
├── checkpoints/
│   ├── vanilla_best.pt
│   └── temporal_best.pt
└── outputs/            ← paper figures go here
```

## Citation

If you use this in your report, cite:

```
[5] F. Locatello et al., "Object-Centric Learning with Slot Attention," NeurIPS 2020.
[7] N. Chung et al., "Rethinking Progression of Memory State in Robotic Manipulation:
    An Object-Centric Perspective," arXiv, Nov. 2025.
```
