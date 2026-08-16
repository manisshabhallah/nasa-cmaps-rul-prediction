"""
src/models/uncertainty.py
--------------------------
UNCERTAINTY QUANTIFICATION (UQ) MODULE

Two types of uncertainty in RUL prediction (Bayesian PHM literature):

1. ALEATORIC UNCERTAINTY — noise in the data itself
   "Even with infinite data I can't predict this exactly"
   → Captured by our Gaussian NLL output (mean + log_var)
   → Cannot be reduced by more data

2. EPISTEMIC UNCERTAINTY — uncertainty in the MODEL WEIGHTS
   "If I had more training data, my model would be more confident"
   → Captured by Monte Carlo Dropout (Gal & Ghahramani, 2016)
   → Can be reduced with more data

HOW MC DROPOUT WORKS:
  During normal inference, Dropout is turned OFF.
  In MC Dropout, we keep Dropout ON at inference time,
  then run the same input through the model T=50 times.
  Each time, different neurons are randomly dropped → different prediction.
  The spread of these 50 predictions = epistemic uncertainty.

  Combined std = sqrt(aleatoric_var + epistemic_var)
  This gives total uncertainty with a physical interpretation.

WHY THIS MATTERS FOR PHM:
  Safety-critical decision: "Should we ground this aircraft?"
  If predicted RUL=30 ± 5 cycles → fairly confident → decide based on it
  If predicted RUL=30 ± 40 cycles → model is unsure → human review needed
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List


def enable_dropout(model: nn.Module):
    """
    Keep Dropout layers active during inference.
    PyTorch normally disables them in model.eval() mode.
    We override this for MC Dropout.
    """
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


def mc_dropout_predict(model: nn.Module,
                       x: torch.Tensor,
                       n_samples: int = 50) -> dict:
    """
    Monte Carlo Dropout inference.

    Runs the model `n_samples` times with dropout active.
    Returns mean prediction + BOTH uncertainty components.

    Args:
        model    : trained CNNTransformerRUL
        x        : input tensor (batch, seq_len, n_features)
        n_samples: number of stochastic forward passes

    Returns dict:
        rul_mean       : (batch,) numpy — mean over MC samples
        epistemic_std  : (batch,) numpy — std of MC means (model uncertainty)
        aleatoric_std  : (batch,) numpy — sqrt of mean variance (data uncertainty)
        total_std      : (batch,) numpy — combined uncertainty
        all_samples    : (n_samples, batch) — all raw MC samples for diagnostics
    """
    model.eval()
    enable_dropout(model)   # keep dropout ON

    mc_means = []
    mc_vars  = []

    with torch.no_grad():
        for _ in range(n_samples):
            output = model(x)
            mean, log_var = output[0], output[1]
            mc_means.append(mean.cpu().numpy())
            mc_vars.append(torch.exp(log_var).cpu().numpy())

    mc_means = np.array(mc_means)    # (n_samples, batch)
    mc_vars  = np.array(mc_vars)     # (n_samples, batch)

    # Epistemic = variance ACROSS the MC predictions
    epistemic_var = mc_means.var(axis=0)

    # Aleatoric = mean of the model's predicted variances
    aleatoric_var = mc_vars.mean(axis=0)

    # Total uncertainty = epistemic + aleatoric (law of total variance)
    total_var = epistemic_var + aleatoric_var

    return {
        "rul_mean"      : mc_means.mean(axis=0),
        "epistemic_std" : np.sqrt(epistemic_var),
        "aleatoric_std" : np.sqrt(aleatoric_var),
        "total_std"     : np.sqrt(total_var),
        "all_samples"   : mc_means,
        "rul_lower_95"  : np.percentile(mc_means, 2.5,  axis=0),
        "rul_upper_95"  : np.percentile(mc_means, 97.5, axis=0),
    }


class ModelEnsemble:
    """
    DEEP ENSEMBLE for uncertainty quantification.

    Train N independent models with different random seeds.
    Each model sees the same data but converges to a different
    local minimum. The disagreement between models = epistemic uncertainty.

    Lakshminarayanan et al. (2017): "Simple and Scalable Predictive
    Uncertainty Estimation using Deep Ensembles" — cited 5000+ times.

    For C-MAPSS: N=5 models, each trained 50 epochs.
    Adds ~5× training time but gives much more reliable uncertainty.
    """

    def __init__(self, models: List[nn.Module]):
        self.models = models
        self.n_models = len(models)

    def predict(self, x: torch.Tensor) -> dict:
        """
        Runs all ensemble members and aggregates.
        """
        all_means = []
        all_vars  = []

        for model in self.models:
            model.eval()
            with torch.no_grad():
                output = model(x)
                mean, log_var = output[0], output[1]
                all_means.append(mean.cpu().numpy())
                all_vars.append(torch.exp(log_var).cpu().numpy())
               

        all_means = np.array(all_means)   # (n_models, batch)
        all_vars  = np.array(all_vars)    # (n_models, batch)

        # Mixture of Gaussians:
        # Combined mean = average of individual means
        combined_mean = all_means.mean(axis=0)

        # Combined variance = mean of variances + variance of means
        # (This is the exact formula for mixture of Gaussians)
        combined_var = (all_vars + all_means**2).mean(axis=0) - combined_mean**2
        combined_var = np.maximum(combined_var, 1e-6)

        return {
            "rul_mean"      : combined_mean,
            "total_std"     : np.sqrt(combined_var),
            "rul_lower_95"  : combined_mean - 1.96 * np.sqrt(combined_var),
            "rul_upper_95"  : combined_mean + 1.96 * np.sqrt(combined_var),
            "individual_preds": all_means,
        }

    def save(self, path_template: str):
        """Save each model: path_template like 'models/ensemble_{i}.pth'"""
        import os
        os.makedirs(os.path.dirname(path_template.format(i=0)), exist_ok=True)
        for i, model in enumerate(self.models):
            torch.save(model.state_dict(), path_template.format(i=i))

    @classmethod
    def load(cls, path_template: str, model_class, model_kwargs: dict,
             n_models: int = 5):
        """Load ensemble from saved checkpoints."""
        models = []
        for i in range(n_models):
            m = model_class(**model_kwargs)
            m.load_state_dict(torch.load(path_template.format(i=i),
                                         map_location="cpu"))
            models.append(m)
        return cls(models)


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
    from src.models.transformer_rul import CNNTransformerRUL

    model = CNNTransformerRUL(n_features=42)
    x = torch.randn(4, 30, 42)

    result = mc_dropout_predict(model, x, n_samples=30)
    print("MC Dropout prediction:")
    for k, v in result.items():
        if k != "all_samples":
            print(f"  {k}: {np.array(v).round(2)}")
