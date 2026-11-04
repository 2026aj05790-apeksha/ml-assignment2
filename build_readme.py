"""Generate README.md from the REAL results (results/metrics.csv, confusion matrices, saved models)."""
import json, joblib, numpy as np, pandas as pd
res = pd.read_csv("results/metrics.csv").set_index("ML Model Name")
cms = {k: np.array(v) for k, v in json.load(open("results/confusion_matrices.json")).items()}
cfg = json.load(open("config.json")); gh = cfg.get("github_url") or "https://github.com/<your-username>/<repo-name>  *(replace after pushing)*"
app = cfg.get("app_url") or "https://<your-app>.streamlit.app  *(replace after deploying)*"
te = pd.read_csv("test_data.csv"); X, y = te.drop(columns="diagnosis"), te["diagnosis"]
wrong = {}
for n in res.index:
    m = joblib.load("model/" + n.lower().replace(" ", "_") + ".joblib"); wrong[n] = set(np.where(m.predict(X) != y)[0])
always = set.intersection(*wrong.values()); anyw = set.union(*wrong.values())
fn = {n: int(c[0, 1]) for n, c in cms.items()}  # actual malignant(0) predicted benign(1)
fp = {n: int(c[1, 0]) for n, c in cms.items()}
r = res
def tbl():
    h = "| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |\n|---|---|---|---|---|---|---|\n"
    return h + "\n".join(f"| {n} | " + " | ".join(f"{r.loc[n, c]:.4f}" for c in ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]) + " |" for n in r.index)
def cv(n): return f"{r.loc[n,'CV Acc (5-fold, train)']:.3f} ± {r.loc[n,'CV Std']:.3f}"
O = {
"Logistic Regression": f"Reaches {r.loc['Logistic Regression','Accuracy']:.2%} accuracy, AUC {r.loc['Logistic Regression','AUC']:.4f} and MCC {r.loc['Logistic Regression','MCC']:.3f} (tied best), with only {fn['Logistic Regression']+fp['Logistic Regression']} errors out of {len(y)} test rows. This works because the malignant/benign classes are almost linearly separable in standardised feature space, so a linear boundary has low bias and, with only 30 inputs, low variance. Train accuracy ({r.loc['Logistic Regression','Train Acc']:.3f}) is close to CV ({cv('Logistic Regression')}), indicating no overfitting. Coefficients are directly interpretable (see feature-importance figure).",
"Decision Tree": f"Weakest model: accuracy {r.loc['Decision Tree','Accuracy']:.2%}, AUC {r.loc['Decision Tree','AUC']:.3f}, MCC {r.loc['Decision Tree','MCC']:.3f}. Train accuracy ({r.loc['Decision Tree','Train Acc']:.3f}) far exceeds test accuracy, a classic high-variance overfitting signature even with max_depth=5 and min_samples_leaf=3, because axis-aligned splits must approximate a diagonal boundary with many small steps and 455 training rows are too few to stabilise them. It missed {fn['Decision Tree']} malignant cases (recall {r.loc['Decision Tree','Recall']:.3f}), clinically the worst outcome. Its coarse probability estimates also pull AUC down. Benefit: fully transparent rules.",
"KNN": f"Accuracy {r.loc['KNN','Accuracy']:.2%} and perfect recall (1.000, zero missed malignant... see confusion matrix: {fn['KNN']} false negatives) with {fp['KNN']} false positives, so precision ({r.loc['KNN','Precision']:.3f}) is the cost. Distance-weighted k=7 on standardised features works because similar tumours cluster in feature space; without scaling it would be dominated by area/perimeter features. Train accuracy is 1.000 (distance weighting memorises training points) while CV is {cv('KNN')}, so the CV standard deviation is the largest of the distance-based models and prediction is slower since it stores all data.",
"Naive Bayes": f"Accuracy {r.loc['Naive Bayes','Accuracy']:.2%}, MCC {r.loc['Naive Bayes','MCC']:.3f} but a strong AUC of {r.loc['Naive Bayes','AUC']:.4f}, meaning it ranks cases well yet its 0.5 cut-off is poorly calibrated. The independence assumption is violated here because radius, perimeter and area (and their 'worst' versions) are correlated above 0.95 (see heatmap); correlated evidence is counted multiple times, making posteriors overconfident. Train ({r.loc['Naive Bayes','Train Acc']:.3f}) and test accuracies are similar, so it suffers from bias rather than variance. It is the fastest model and needs no tuning.",
"Support Vector Machine": f"Ties with Logistic Regression on every test metric (accuracy {r.loc['Support Vector Machine','Accuracy']:.2%}, AUC {r.loc['Support Vector Machine','AUC']:.4f}, MCC {r.loc['Support Vector Machine','MCC']:.3f}). The RBF kernel maximises the margin and handles any mild non-linearity, and scaling makes gamma='scale' meaningful. CV accuracy ({cv('Support Vector Machine')}) is slightly below logistic regression, suggesting the extra flexibility buys nothing on this nearly linear problem while costing training time and interpretability.",
}
O["KNN"] = O["KNN"].replace("perfect recall (1.000, zero missed malignant... see confusion matrix: ", "perfect recall (1.000) with ").replace(" false negatives) with ", " missed malignant cases and ")
win = (f"**Logistic Regression** (with SVM as a statistical tie). LR and SVM have identical test accuracy ({r.loc['Logistic Regression','Accuracy']:.2%}), MCC ({r.loc['Logistic Regression','MCC']:.3f}) and recall, "
       f"but LR has the higher 5-fold CV accuracy ({cv('Logistic Regression')} vs {cv('Support Vector Machine')}), the highest AUC, trains in milliseconds and is interpretable. "
       f"We rank by MCC and recall rather than accuracy alone, because MCC is robust to the 63/37 class imbalance and a false negative (missed malignancy) is the costliest error. "
       f"KNN has the best recall but lower precision/MCC. With only {len(y)} test rows, one sample moves accuracy by {1/len(y):.2%}, so LR vs SVM differences are not significant.")
fig = lambda f, c: f"![{c}](figures/{f}.png)\n*{c}*\n"
md = f"""# 🩺 Breast-Cancer Diagnosis – Five-Classifier Comparison & Streamlit Deployment
**BITS Pilani WILP · M.Tech (AIML) · Machine Learning · Assignment 2**

## a. Problem statement
Early, accurate discrimination between **malignant** and **benign** breast tumours from fine-needle-aspirate (FNA) cell-nucleus measurements reduces unnecessary biopsies and prevents missed cancers.
The task is a **binary classification** problem: build, evaluate and compare five classifiers (Logistic Regression, Decision Tree, KNN, Naive Bayes, SVM) on one dataset using six metrics
(Accuracy, AUC, Precision, Recall, F1, MCC), then deploy an interactive Streamlit app that evaluates the *saved* models on an uploaded test CSV without retraining.

## b. Dataset description
| Item | Detail |
|---|---|
| **Source** | Breast Cancer Wisconsin (Diagnostic) – UCI Machine Learning Repository (Wolberg, Street & Mangasarian, 1995), loaded via `sklearn.datasets.load_breast_cancer` |
| **Instances** | 569 (train 455 / test 114, stratified 80:20, seed 42) |
| **Features** | 30 numeric (>= 12 required): 10 nucleus measurements × {{mean, standard error, worst}} |
| **Target** | `diagnosis` – 0 = malignant (212, 37.3 %), 1 = benign (357, 62.7 %) |
| **Class balance** | Moderately imbalanced (1 : 1.68) → MCC and recall reported alongside accuracy |
| **Missing values** | None |
| **Pre-processing** | `StandardScaler` inside each sklearn `Pipeline` (fit on training data only – no leakage); the saved `.joblib` therefore contains the complete preprocessing |

The 10 base measurements: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension.
Many are strongly collinear (radius–perimeter–area r > 0.95), which matters for Naive Bayes (see observations).

{fig('01_class_distribution','Figure 1 – Class distribution')}
{fig('02_correlation_heatmap','Figure 2 – Feature correlation heatmap: blocks of highly correlated size features')}

## c. GitHub Repository Link
{gh}

```
project-folder/
├── app.py                 # Streamlit app (upload CSV → choose model → metrics, confusion matrix, report, comparison)
├── train.py               # trains 5 models, computes metrics, saves models + figures
├── build_readme.py        # regenerates this README from real results
├── requirements.txt
├── test_data.csv          # 114-row hold-out test set (30 features + diagnosis)
├── full_dataset.csv       # complete 569-row dataset (for reference)
├── model/                 # 5 saved pipelines (.joblib)
├── results/               # metrics.csv, confusion_matrices.json
├── figures/               # all plots used below
└── agents/                # grader agent + improvement loop (development tooling)
```
**Live Streamlit app:** {app}

### How to run / reproduce
```bash
pip install -r requirements.txt
python train.py            # retrains, rewrites model/, results/, figures/, test_data.csv
streamlit run app.py       # launches the evaluation app
```

## d. Models used
All five models are trained on the **same** 455-row training split and evaluated on the **same** 114-row hold-out test set.
Hyper-parameters: LR (C=1), Decision Tree (max_depth=5, min_samples_leaf=3), KNN (k=7, distance-weighted), Gaussian NB (defaults), SVM (RBF, C=1, gamma='scale').

### Comparison table (test set)
{tbl()}

### Generalisation check (5-fold stratified CV on training data vs train vs test)
| ML Model Name | Train Acc | 5-fold CV Acc | Test Acc |
|---|---|---|---|
""" + "\n".join(f"| {n} | {r.loc[n,'Train Acc']:.4f} | {cv(n)} | {r.loc[n,'Accuracy']:.4f} |" for n in r.index) + f"""

{fig('06_metric_comparison','Figure 3 – Metric comparison across models')}
{fig('04_roc_curves','Figure 4 – ROC curves (test set)')}
{fig('05_confusion_matrices','Figure 5 – Confusion matrices (class 0 = malignant)')}
{fig('08_overfit_check','Figure 6 – Train vs CV vs test accuracy')}
{fig('09_feature_importance','Figure 7 – Feature importance (Decision Tree) and coefficients (Logistic Regression)')}

### Observations on model performance
| ML Model Name | Observation about model performance |
|---|---|
""" + "\n".join(f"| {n} | {O[n]} |" for n in r.index) + f"""
| **Overall, Winner for your dataset?** | {win} |

### Error analysis
{len(anyw)} of {len(y)} test samples are misclassified by at least one model and **{len(always)}** by all five – these are genuinely borderline tumours whose measurements overlap the other class.
Most of the accuracy gap between models therefore comes from a small set of ambiguous cases, which is why MCC/recall and CV are used in addition to a single test split.

## Streamlit application features
CSV upload (test data only) · model dropdown · six metric cards · confusion matrix + classification report · ROC curve · adjustable decision threshold ·
side-by-side comparison of all five saved models on the uploaded data · prediction download · input validation (missing columns / non-binary target).
Models are loaded with `joblib`; **no training happens in the app**.

## Limitations
* Single public dataset with 569 rows; a 114-row test set gives wide confidence intervals (one sample ≈ 0.88 % accuracy), so LR vs SVM differences are not statistically significant.
* Hyper-parameters were set by reasoned defaults and light CV, not an exhaustive search; data come from one institution, so external generalisation is unverified.
* Models are decision-support tools, not diagnostic devices.

## Future work
* Nested cross-validation with grid/Bayesian search, probability calibration (Platt/isotonic) – especially for Naive Bayes.
* Ensembles (Random Forest, Gradient Boosting, stacking), SHAP explanations, external validation on another cohort.
* Cost-sensitive threshold selection to explicitly minimise false negatives.
"""
open("README.md", "w", encoding="utf-8").write(md); print("README words:", len(md.split()))
