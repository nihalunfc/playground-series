## 📊 Experiment Results Summary
(Dated: 21 Feb 2026, 6 AM EST)

| Trial ID | Strategy Description | Public Score | Outcome |
| :--- | :--- | :--- | :--- |
| **01** | XGBoost Baseline | 0.95309 | Baseline established |
| **04** | Optuna Hyperparameter Tuning | 0.95548 (Local) | Found optimal parameters |
| **07** | **Stacking Ensemble (XGB + LGBM)** | **0.95370** | **🏆 Best Model (Selected)** |
| **08** | Pseudo-Labeling Augmentation | 0.95368 | Slight noise introduced |
| **11** | Super Stack (Fully Optimized) | 0.95364 | Over-optimization (loss of diversity) |
| **12** | Diverse Trio (+Random Forest) | 0.95309 | Random Forest reduced precision |
| **13** | Grand Hybrid Blend | 0.95354 | Complexity reduced generalization |

**Conclusion:** The Stacking Ensemble (Trial 07) provided the robust generalization required for the test set, proving that architectural balance is more effective than raw complexity or aggressive data augmentation for this specific dataset.
