# 📊 Computer Convergence Application Assignments.. by Victini00 (2025)

This repository contains completed assignments and implementations for the **Computer Convergence Applications (CCA)** course.  
Each assignment focuses on **practical machine learning techniques**;

1. Dimensionality reduction
2. Tree based methods
3. PEFT - LoRA

All experiments are implemented in **Python (Jupyter Notebook)** with fixed random seeds for reproducibility.

⭐️ **The form of the provided skeleton codes were not modified as much as possible.**

---

## 🧪 Assignment #1 — Dimensionality Reduction & Feature Analysis

**Topics**  
Logistic Regression, L1/L2 Regularization, PCA, ICA, CCA, t-SNE

**Dataset**  
- Breast Cancer Wisconsin Dataset (`sklearn.datasets.load_breast_cancer`)

**Main Tasks**
- Logistic Regression with **L2 regularization**
  - Test accuracy and ROC-AUC evaluation
  - Analysis of top-5 features by absolute coefficient magnitude
- Logistic Regression with **L1 regularization (Lasso)**
  - Identification of non-zero coefficients
  - Comparison with L2-selected features
- **Principal Component Analysis (PCA)**
  - 2D projection and explained variance analysis
- **Independent Component Analysis (ICA)**
  - Comparison with PCA in terms of separation characteristics
- **Canonical Correlation Analysis (CCA)**
  - Multi-view feature correlation analysis
- **t-SNE**
  - Visualization with different perplexity values (10, 30, 50)

**Highlights**
- Clear comparison between feature-based and projection-based representations
- Visualization-driven understanding of dimensionality reduction methods

---

## 🌳 Assignment #2 — Tree-Based Models for Drug Toxicity Prediction

**Topics**  
Random Forest, XGBoost, Regularization Analysis

**Dataset**
- Drug-Induced Liver Injury (DILI) dataset from **Therapeutics Data Commons (TDC)**
- Molecular features extracted using **RDKit descriptors**

**Main Tasks**
- **Random Forest**
  - Performance comparison across different `n_estimators`
  - Evaluation of `gini` vs `entropy` splitting criteria
- **XGBoost**
  - Accuracy analysis for different **L2 regularization (`reg_lambda`)** values
  - Accuracy analysis for different **L1 regularization (`alpha`)** values
- **Regularization Interpretation**
  - Determining which hyperparameter better controls L1 regularization effects

**Highlights**
- Controlled hyperparameter sweeps under fixed experimental settings
- Quantitative comparison of regularization strategies
- Practical insights for applying tree-based models to clinical datasets

---

## 🧬 Assignment #3 — Parameter-Efficient Fine-Tuning with LoRA

**Topics**  
Pre-trained Protein Language Models, LoRA, Efficient Fine-Tuning

**Dataset**
- Virulence Factor Classification Dataset (FASTA format)
- Binary classification (virulent vs non-virulent)

**Main Tasks**

### 1. Base Model vs LoRA (r = 8)
- Full fine-tuning of a pre-trained protein language model
- LoRA fine-tuning with rank `r = 8`
- Comparison of:
  - Number of trainable parameters
  - Validation and test **Accuracy** and **F1-score**

### 2. LoRA Rank Sweep (r = 3–10)
- Performance evaluation across different LoRA ranks
- Recording validation and test metrics for each rank
- Identification of a potential performance–efficiency trade-off point

### 3. Result Interpretation
- When LoRA becomes competitive with full fine-tuning
- Trade-offs between expressiveness and parameter efficiency
- Stability and limitations observed at different ranks

**Highlights**
- Empirical demonstration of LoRA’s parameter efficiency
- Clear performance trends with respect to rank size
- Practical discussion on the applicability of LoRA for biological sequence tasks
---

## ⚙️ Reproducibility

- Random seed fixed to **42** for all experiments
- Consistent train/validation/test splits within each assignment
- All notebooks run end-to-end using **Restart & Run All**


## 📌 Notes

- This repository is intended for **educational and academic purposes only**.