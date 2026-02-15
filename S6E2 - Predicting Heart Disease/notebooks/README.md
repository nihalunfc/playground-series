# 📂 Experiment Log & Notebooks
**Project:** Heart Disease Risk Prediction (Kaggle S6E2)

This directory contains the complete experimental history of the project. Each notebook represents a specific hypothesis, strategy, or architectural pivot. Below is a detailed breakdown of the models used, their theoretical advantages, and the observed drawbacks during this competition.

---

## 🏛️ Phase 1: The Foundation

### `01_XGBoost_Baseline.ipynb`
**Model:** Extreme Gradient Boosting (XGBoost) with default parameters.
* **Description:** A raw implementation of XGBoost to establish a performance baseline.
* **✅ Advantages:** XGBoost is the industry standard for tabular data due to its speed and ability to handle missing values natively. It uses gradient descent to minimize loss, often outperforming Random Forests on structured data.
* **❌ Drawbacks:** Without tuning, it is prone to overfitting on small datasets and may not capture the optimal tree depth for the specific signal-to-noise ratio of the data.
* **Outcome:** Established a baseline score of `0.95309`.

---

## ⚙️ Phase 2: Optimization

### `04_Hyperparameter_Optimization_Optuna.ipynb`
**Technique:** Bayesian Optimization using Optuna.
* **Description:** A search algorithm designed to find the global maximum of the objective function (AUC) by tuning hyperparameters like `learning_rate`, `max_depth`, and `subsample`.
* **✅ Advantages:** Unlike Grid Search (which tries every combination blindy) or Random Search, Bayesian Optimization "learns" from previous trials. It focuses computational power on promising areas of the hyperparameter space.
* **❌ Drawbacks:** It is computationally expensive and can lead to "over-optimization," where the model becomes too specialized to the validation set and loses generalization power on the test set.
* **Outcome:** Found a "shallow tree" configuration (`max_depth: 4`) that boosted validation AUC to `0.95548`.

### `05_Optimized_XGBoost_Submission.ipynb`
**Model:** XGBoost (Optuna-Tuned).
* **Description:** The deployment of the parameters found in Trial 04 on the full training dataset.
* **✅ Advantages:** Maximized the predictive power of a single algorithm.
* **❌ Drawbacks:** Single models, no matter how tuned, have a "performance ceiling" because they cannot correct their own intrinsic biases.
* **Outcome:** Strong performance, but hit a plateau.

---

## 🏗️ Phase 3: Ensemble Architectures

### `06_Master_Blend_Ensemble.ipynb`
**Technique:** Weighted Averaging (Blending).
* **Description:** A linear combination of the Optimized XGBoost (70%) and a standard LightGBM (30%).
* **✅ Advantages:** Reduces variance. If XGBoost makes a mistake on a specific row, LightGBM might correct it.
* **❌ Drawbacks:** The weights (0.7 / 0.3) were manually assigned based on intuition ("XGBoost is better, so give it more weight"). This is unscientific and suboptimal.
* **Outcome:** Failed to improve significantly over the single model.

### `07_Advanced_Stacking_Ensemble.ipynb` (🏆 The Winner)
**Technique:** Stacked Generalization (Stacking).
* **Description:** Uses a **Meta-Learner** (Logistic Regression) to learn how to combine the predictions of the Base Learners (Optimized XGBoost + Standard LightGBM).
* **✅ Advantages:** The Meta-Learner dynamically assigns importance to models based on the input data. It learns *when* to trust XGBoost and *when* to trust LightGBM, effectively automating the weighting process.
* **❌ Drawbacks:** Adds complexity to the pipeline and doubles the inference time.
* **Outcome:** **Best Public Score: 0.95370.**

---

## 🧪 Phase 4: Experimental Architectures

### `08_Pseudo_Labeling_Experiment.ipynb`
**Technique:** Semi-Supervised Learning (Pseudo-Labeling).
* **Description:** Used the model's high-confidence predictions on the Test set to augment the Training set.
* **✅ Advantages:** Increases the effective size of the training data and helps the model align with the distribution of the test set (Domain Adaptation).
* **❌ Drawbacks:** High risk of "Confirmation Bias." If the model is wrong about a prediction and we feed that back in as "truth," the model reinforces its own error.
* **Outcome:** Slight overfitting (`0.95368`).

### `09_Trinity_Stack_Ensemble.ipynb`
**Model:** Stacking with **CatBoost** added.
* **Description:** Added CatBoost (Categorical Boosting) to the stack to create a 3-model diversity.
* **✅ Advantages:** CatBoost uses "Ordered Boosting" and handles categorical features differently than XGB/LGBM, theoretically adding a unique perspective.
* **❌ Drawbacks:** In this specific dataset, the CatBoost model was not sufficiently tuned, introducing noise rather than signal.
* **Outcome:** Score dropped (`0.95358`).

### `10_LightGBM_Optimization_Optuna.ipynb`
**Technique:** Bayesian Optimization for LightGBM.
* **Description:** Ran a dedicated Optuna search to find the perfect parameters for LightGBM.
* **✅ Advantages:** Ensured the LightGBM component was as strong as the XGBoost component.
* **Outcome:** Identified that LightGBM also preferred shallow trees (`max_depth: 3`).

### `11_Super_Stack_Ensemble.ipynb`
**Technique:** Fully Optimized Stacking.
* **Description:** Combined Optimized XGBoost + Optimized LightGBM.
* **✅ Advantages:** Every component of the stack was mathematically perfect.
* **❌ Drawbacks (The "Diversity Paradox"):** By optimizing both models for the exact same metric (AUC), they converged on the *same* solution. They lost their "diversity of thought," making the ensemble less effective than the "Optimized + Standard" mix in Trial 07.
* **Outcome:** Score dropped (`0.95364`).

### `12_Diverse_Trio_Ensemble.ipynb`
**Model:** Stacking with **Random Forest**.
* **Description:** Replaced CatBoost with a Random Forest Classifier.
* **✅ Advantages:** Random Forest uses Bagging (Bootstrap Aggregating) instead of Boosting. It builds independent trees, making it structurally very different from XGBoost/LightGBM. Great for stability.
* **❌ Drawbacks:** Random Forests are generally less precise than Gradient Boosting on clean tabular data. It acted as a "weak link," dragging down the overall precision.
* **Outcome:** Score dropped (`0.95309`).

### `13_Grand_Hybrid_Ensemble.ipynb`
**Technique:** Hybrid Blending.
* **Description:** A weighted average of the **Stacking Model (Trial 07)** and the **Pseudo-Labeled Model (Trial 08)**.
* **✅ Advantages:** Attempts to capture the structural benefits of Stacking and the data augmentation benefits of Pseudo-Labeling.
* **❌ Drawbacks:** Complexity overload. The small gains from Pseudo-Labeling were likely just noise, so blending them diluted the purity of the Stacking model.
* **Outcome:** `0.95354`.

---

## 🔑 Key Learnings
1.  **Architecture Wins:** The jump from single models to Stacking (Trial 07) provided the biggest reliable gain.
2.  **Diversity > Perfection:** An ensemble of one "smart" model (Optimized XGB) and one "dumb" model (Standard LGBM) outperformed two "smart" models (Super Stack). The difference in their errors is what allows the Meta-Learner to work.
3.  **Complexity has a Cost:** Adding more models (CatBoost, Random Forest) or complex techniques (Pseudo-Labeling) does not guarantee better results if the underlying signal is already well-captured.
