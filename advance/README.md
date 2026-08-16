# Advanced PHM Framework — NASA C-MAPSS
## M.Tech Data Engineering | Manisha Bhalla (M25DE1050)
### PhD-Level End-to-End Prognostics & Health Management

---

## What Makes This "PhD-Level"

| Feature | Basic Version | This (Advanced) |
|---------|--------------|-----------------|
| Model | 2-layer LSTM | CNN + Dual-Attention Transformer |
| Training data | FD001 only (~17K samples) | FD001-FD004 joint (~100K after augmentation) |
| Output | Single RUL number | RUL + uncertainty interval |
| Uncertainty type | None | Epistemic + Aleatoric (decomposed) |
| Loss function | MSE | Gaussian Negative Log-Likelihood |
| Training | Standard | Curriculum + Multi-task learning |
| LR schedule | ReduceLROnPlateau | Cosine Annealing with Warm Restarts |
| Evaluation | RMSE, MAE | RMSE, MAE, NASA Score, α-λ, PH, ECE |
| Literature grounding | 2015 | 2023–2025 SOTA |

---

## Project Structure

```
phm_advanced/
│
├── src/
│   ├── data/
│   │   └── augmentation.py      ← Cross-dataset fusion + noise + jitter
│   │
│   ├── models/
│   │   ├── transformer_rul.py   ← CNN-Transformer with dual-axis attention
│   │   └── uncertainty.py       ← MC Dropout + Deep Ensemble
│   │
│   └── training/
│       ├── train_advanced.py    ← Curriculum + multi-task training
│       └── evaluate_advanced.py ← α-λ, PH, ECE, critical zone RMSE
│
├── dashboard/
│   └── app_advanced.py          ← Streamlit (6 pages, uncertainty UI)
│
├── run_pipeline_advanced.py     ← Master runner
└── requirements.txt
```

**NOTE:** The bronze/silver/gold data pipeline is shared with the basic project.
Copy or symlink `src/bronze_layer.py`, `src/silver_layer.py`, `src/gold_layer.py`
from the basic project, OR run both from the same folder.

---

## How to Run

### Step 1 — Run the basic pipeline for ALL four datasets first

```bash
# In your phm_project/ folder (basic pipeline)
python run_pipeline.py --dataset FD001 --skip_shap
python run_pipeline.py --dataset FD002 --skip_shap
python run_pipeline.py --dataset FD003 --skip_shap
python run_pipeline.py --dataset FD004 --skip_shap
```

This creates the gold data (sliding windows) for each dataset.

### Step 2 — Copy advanced files into the same project folder

Copy these into your phm_project/:
- `src/data/augmentation.py`
- `src/models/transformer_rul.py`
- `src/models/uncertainty.py`
- `src/training/train_advanced.py`
- `src/training/evaluate_advanced.py`
- `dashboard/app_advanced.py`
- `run_pipeline_advanced.py`

### Step 3 — Run the advanced pipeline

```bash
# Full run: all 4 datasets, all techniques (takes 10–20 min on CPU)
python run_pipeline_advanced.py

# Quick test: FD001 only, no augmentation
python run_pipeline_advanced.py --datasets FD001 --no_augmentation --epochs 30

# Custom hyperparameters
python run_pipeline_advanced.py \
    --datasets FD001 FD002 FD003 FD004 \
    --epochs 80 \
    --batch_size 512 \
    --d_model 64 \
    --n_heads 4 \
    --n_layers 2
```

### Step 4 — Launch advanced dashboard

```bash
streamlit run dashboard/app_advanced.py
```

---

## Architecture Deep Dive

### CNN-Transformer with Dual-Axis Attention

```
Input: (batch, 30 cycles, 42 features)
         ↓
[Conv1D × 2]     ← local pattern extraction (adjacent cycle spikes)
         ↓
[Positional Encoding]   ← sinusoidal — tells model which cycle is which
         ↓
[Sensor Attention]      ← which sensors matter at each timestep?
         ↓
[Temporal Transformer]  ← which timesteps matter overall?
  └── 2 layers of Multi-Head Self-Attention + FFN + LayerNorm
         ↓
[Weighted Aggregation]  ← learned query decides timestep importance
         ↓
[Dense 32 + GELU]
         ↓
    ┌─────────┴──────────┐
[mean head]          [log_var head]
pred_RUL (float)     uncertainty (float)
```

### Why Dual Attention?

Standard Transformer = attends across TIME only, treats all sensors equally.

Sensor Attention:  "At cycle 25, HPC temperature matters 3× more than fan speed"
Temporal Attention: "Cycles 25–30 matter more than cycles 1–10 for this prediction"

Both learned jointly from data. This is from Fan et al. (2024), Sensors journal.

### Data Augmentation Strategy

Base: FD001=100 engines, FD002=260, FD003=100, FD004=249 → 709 engines total
After windowing: ~250,000 sequences
After noise augmentation: ~500,000 sequences
After window jitter (steps=2): ~1,500,000 sequences

In practice we use a manageable subset. The key insight is that cross-dataset
training teaches the model that engines from different operating conditions
all share the same underlying degradation physics — it just looks different
on the surface.

### Uncertainty Quantification

**Aleatoric** (data noise):
- Model outputs log(σ²) alongside the RUL mean
- Trained with Gaussian NLL: forces model to learn its own error bounds
- High aleatoric σ → sensor readings are noisy/ambiguous for this engine

**Epistemic** (model uncertainty):
- MC Dropout: keep dropout ON at inference, run 50 forward passes
- Variance of the 50 predictions = how much the model "disagrees with itself"
- High epistemic σ → this engine is very different from training engines

**Total σ** = √(aleatoric² + epistemic²)

### Curriculum Learning Schedule

Epoch 1–15:  train only on sequences where RUL > 40 (clear healthy/degrading)
Epoch 16+:   train on all sequences including near-failure (RUL < 20)

Why: near-failure sensor patterns are hard to distinguish without context.
Learning the full degradation trajectory first gives the model a foundation.

### Evaluation Metrics

| Metric | What it measures | Formula |
|--------|-----------------|---------|
| RMSE | Overall prediction accuracy | √(mean((ŷ-y)²)) |
| Critical RMSE | Accuracy when RUL ≤ 30 (safety zone) | RMSE filtered to RUL≤30 |
| NASA Score | Asymmetric — late prediction penalized more | Σ exp(d/10)-1 or exp(-d/13)-1 |
| α-λ accuracy | At λ=60% of life, within ±20%? | Fraction of engines meeting criterion |
| ECE | Are uncertainty intervals statistically honest? | |coverage - confidence| |
| Mean σ | Average prediction uncertainty | mean(σ across test set) |

---

## References

1. Ramasso & Saxena (2015). Review of Algorithmic Approaches on CMAPSS.
   NASA/TM-2015-218764 — foundation of C-MAPSS benchmark understanding

2. Fan, Li & Chang (2024). Two-Stage Attention-Based Hierarchical Transformer.
   Sensors 24(3), 824. DOI: 10.3390/s24030824

3. Sharma (2024). Uncertainty-Aware Deep Learning for RUL Prediction.
   arXiv:2511.19124

4. Lakshminarayanan et al. (2017). Simple and Scalable Predictive Uncertainty
   Estimation using Deep Ensembles. NeurIPS 2017.

5. Gal & Ghahramani (2016). Dropout as a Bayesian Approximation. ICML 2016.

6. Bengio et al. (2009). Curriculum Learning. ICML 2009.
