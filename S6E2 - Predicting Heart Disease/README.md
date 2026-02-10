# Heart Disease Prediction (Playground S6E2)

## The Story So Far
I have stepped into the medical domain for the second episode of the 2026 season. My goal is to predict the likelihood of heart disease using a synthetic dataset. This isn't just about a high score; it's about understanding the delicate balance of physiological markers like cholesterol, age, and blood pressure.

### Methodology
1. **The Setup:** I am utilizing **VS Code** for my local development environment to maintain full control over my versioning.
2. **The Metric:** The competition is judged on **AUC-ROC** (Area Under the Receiver Operating Characteristic Curve).
    * *In simple terms:* I'm not just saying "Yes" or "No" to heart disease. I'm ranking every patient by their probability of risk.
3. **The Strategy:** I am currently performing a deep dive into the feature distributions to see if the "synthetic" nature of the data has introduced any strange patterns I can exploit.

### Current Progress & Log 
* **Initial Commit:** Structured the project and downloaded the 630,000-row training set.
* **Next Step:** Perform a comprehensive EDA to visualize how `Age` and `Cholesterol` interact.

---
*This project utilizes datasets provided by Kaggle and leverages open-source ML frameworks.*
