# Heart Disease Risk Prediction | Kaggle Playground S6E2
### Advanced Ensemble Learning Case Study

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat&logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-Optimized-green?style=flat&logo=xgboost)
![LightGBM](https://img.shields.io/badge/LightGBM-Stacking-orange?style=flat)
![Status](https://img.shields.io/badge/Kaggle%20Rank-Top%20400-gold?style=flat&logo=kaggle)

## 📌 Executive Summary
This project was developed as part of my Master's coursework in Data Analytics to explore advanced techniques in **binary classification** and **ensemble learning**. Participating in the Kaggle Playground Series (S6E2), my objective was to predict the risk of heart disease using a synthetic medical dataset.

Rather than relying on a single algorithm, I engineered a 13-stage experimental pipeline to systematically improve model performance. The final solution—a **Stacked Generalization Ensemble**—achieved a Public Leaderboard AUC of **0.95370**, securing a rank within the top tier of competitors (Rank ~367).

---

## 🔬 Methodology & Experimentation
My approach focused on iterative refinement, moving from baseline models to complex architectures to identify the optimal balance between bias and variance.

### 1. Exploratory Data Analysis (EDA) & Baseline
I began by establishing a robust baseline using **XGBoost**. Initial feature engineering efforts, such as creating interaction terms (e.g., `Age * SystolicBP`), yielded marginal gains, suggesting that the underlying signal was strong but required sophisticated architectural handling rather than extensive feature manipulation.
* **Baseline Score:** `0.95309`

### 2. Bayesian Hyperparameter Optimization
To move beyond manual tuning, I integrated **Optuna** for Bayesian Optimization. I conducted 30 automated trials to maximize the Area Under the Curve (AUC).
* **Key Insight:** The optimization process revealed that shallower trees (`max_depth: 4`) combined with a specific learning rate (`0.052`) generalized significantly better than deeper, more complex trees, raising my local validation score to `0.95548`.

### 3. Architectural Breakthrough: Stacking
When single-model performance plateaued, I implemented **Stacked Generalization (Stacking)**.
* **Level 0 (Base Learners):** I combined my Optuna-optimized XGBoost (for high precision) with a Standard LightGBM (for architectural diversity).
* **Level 1 (Meta-Learner):** A Logistic Regression model was used to weigh the predictions of the base learners dynamically.
* **Outcome:** This architecture effectively mitigated individual model biases, resulting in my peak performance of `0.95370`.

### 4. Advanced Experiments (The "Learning" Phase)
I attempted to push performance further using experimental techniques, which provided valuable lessons in overfitting and model diversity:
* **Pseudo-Labeling:** Augmenting the training set with high-confidence test predictions introduced slight noise, dropping the score to `0.95368`.
* **The "Trinity" Stack:** Adding **CatBoost** to the ensemble increased variance, highlighting that adding more models does not automatically equal better performance.
* **Grand Hybrid:** A weighted blend of Stacking and Pseudo-Labeling (`0.95354`) proved too complex, confirming that a simpler, robust stack was the superior strategy.

---

## 📊 Performance Log

| Trial ID | Strategy Description | Public Score (AUC) | Outcome |
| :--- | :--- | :--- | :--- |
| **01** | XGBoost Baseline | 0.95309 | Established baseline |
| **04** | Optuna Optimization | 0.95548 (Local) | Identified optimal hyperparameters |
| **07** | **Stacking Ensemble (XGB + LGBM)** | **0.95370** | **🏆 Best Model (Selected)** |
| **08** | Pseudo-Labeling Augmentation | 0.95368 | Slight overfitting observed |
| **11** | Super Stack (Fully Tuned) | 0.95364 | Loss of ensemble diversity |
| **13** | Grand Hybrid Blend | 0.95354 | Diminishing returns on complexity |

---

## 🛠️ Technical Stack
* **Languages:** Python 3.10+
* **Data Manipulation:** `pandas`, `numpy`
* **Machine Learning:** `scikit-learn`, `xgboost`, `lightgbm`, `catboost`
* **Optimization:** `optuna` (Bayesian Hyperparameter Optimization)
* **Visualization:** `matplotlib`, `seaborn`
* **Environment:** Google Colab Pro (NVIDIA T4 GPU Acceleration)

---

## 🔮 Future Scope
To further improve predictive accuracy and aim for the Top 100, future iterations of this project would focus on:
1.  **Deep Data Cleaning:** Investigating biologically implausible zero-values in `Cholesterol` and `RestingBP` identified during EDA.
2.  **Adversarial Validation:** Ensuring strict alignment between training and test distributions to prevent leakage.
3.  **Neural Networks:** Integrating a PyTorch-based **TabNet** or MLP to introduce non-tree-based diversity into the stack.

---

# To retrain the model and generate a submission file:
python src/train_model.py
*Master's Student, Data Analytics*
