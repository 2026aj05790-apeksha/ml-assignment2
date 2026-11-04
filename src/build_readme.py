"""Generate README.md from the reproducible project artifacts."""
from pathlib import Path
import json, joblib, numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
res = pd.read_csv(ROOT / "artifacts/results/metrics.csv").set_index("ML Model Name")
with open(ROOT / "artifacts/results/confusion_matrices.json") as handle:
    cms = {k: np.array(v) for k, v in json.load(handle).items()}
with open(ROOT / "config/config.json") as handle:
    cfg = json.load(handle)
gh = cfg.get("github_url") or "https://github.com/<your-username>/<repo-name>  *(replace after pushing)*"
app = cfg.get("app_url") or "https://<your-app>.streamlit.app  *(replace after deploying)*"
te = pd.read_csv(ROOT / "artifacts/data/test_data.csv"); X, y = te.drop(columns="diagnosis"), te["diagnosis"]
wrong = {}
for n in res.index:
    m = joblib.load(ROOT / "artifacts/models" / (n.lower().replace(" ", "_") + ".joblib")); wrong[n] = set(np.where(m.predict(X) != y)[0])
always = set.intersection(*wrong.values()); anyw = set.union(*wrong.values())
missed_cancer = {n: int(c[1, 0]) for n, c in cms.items()}  # malignant predicted benign
false_alarm = {n: int(c[0, 1]) for n, c in cms.items()}  # benign predicted malignant
diagnostic = {}
for n, c in cms.items():
    tn, fp, fn, tp = c.ravel()  # clinical labels: 0 = benign, 1 = malignant
    diagnostic[n] = {
        "Balanced Accuracy": (tp / (tp + fn) + tn / (tn + fp)) / 2,
        "Malignant Sensitivity": tp / (tp + fn),
        "Benign Specificity": tn / (tn + fp),
        "Malignant FNR": fn / (tp + fn),
        "Malignant correct": int(tp),
        "Malignant missed": int(fn),
        "Benign flagged": int(fp),
        "Benign correct": int(tn),
    }
r = res
def tbl():
    h = "| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |\n|---|---|---|---|---|---|---|\n"
    return h + "\n".join(f"| {n} | " + " | ".join(f"{r.loc[n, c]:.4f}" for c in ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]) + " |" for n in r.index)
def cv(n): return f"{r.loc[n,'CV Acc (5-fold, train)']:.3f} ± {r.loc[n,'CV Std']:.3f}"
knn_malignant_sensitivity = diagnostic["KNN"]["Malignant Sensitivity"]
knn_malignant_missed = missed_cancer["KNN"]
knn_false_alarm = false_alarm["KNN"]
knn_benign_specificity = diagnostic["KNN"]["Benign Specificity"]
O = {
"Logistic Regression": f"Logistic Regression records {r.loc['Logistic Regression','Accuracy']:.2%} accuracy, AUC {r.loc['Logistic Regression','AUC']:.4f}, and MCC {r.loc['Logistic Regression','MCC']:.3f}. It makes {missed_cancer['Logistic Regression']} missed-cancer error and {false_alarm['Logistic Regression']} benign false alarm on the 114-row test set. Its train score ({r.loc['Logistic Regression','Train Acc']:.3f}) is close to the cross-validation result ({cv('Logistic Regression')}), while the linear coefficients provide useful directionality for interpretation.",
"Decision Tree": f"The Decision Tree has the lowest test performance: {r.loc['Decision Tree','Accuracy']:.2%} accuracy, AUC {r.loc['Decision Tree','AUC']:.3f}, and MCC {r.loc['Decision Tree','MCC']:.3f}. Its training accuracy ({r.loc['Decision Tree','Train Acc']:.3f}) is much higher than its test accuracy, indicating variance and overfitting. It misses {missed_cancer['Decision Tree']} malignant cases, the largest missed-cancer count, although its rule structure remains easy to explain.",
"KNN": f"KNN reaches {r.loc['KNN','Accuracy']:.2%} accuracy and malignant sensitivity of {knn_malignant_sensitivity:.3f}. It misses {knn_malignant_missed} malignant cases and produces {knn_false_alarm} benign false alarms, giving benign specificity of {knn_benign_specificity:.3f}. Standardisation is essential for distance calculations; its training score of 1.0000 is higher than its 5-fold CV score ({cv('KNN')}), showing that memorisation does not transfer perfectly.",
"Naive Bayes": f"Naive Bayes obtains {r.loc['Naive Bayes','Accuracy']:.2%} accuracy and MCC {r.loc['Naive Bayes','MCC']:.3f}, while its AUC is relatively strong at {r.loc['Naive Bayes','AUC']:.4f}. The gap suggests useful ranking but less reliable threshold decisions. Highly correlated radius, perimeter, and area variables conflict with the independence assumption, and the similar train ({r.loc['Naive Bayes','Train Acc']:.3f}) and test scores point more to bias than overfitting.",
"Support Vector Machine": f"The RBF SVM is effectively tied with Logistic Regression: accuracy {r.loc['Support Vector Machine','Accuracy']:.2%}, AUC {r.loc['Support Vector Machine','AUC']:.4f}, and MCC {r.loc['Support Vector Machine','MCC']:.3f}. Its CV accuracy ({cv('Support Vector Machine')}) is below Logistic Regression's, so the additional non-linear flexibility does not improve this split despite the strong test result.",
}
win = (f"**Logistic Regression** (with SVM as a statistical tie). LR and SVM have identical test accuracy ({r.loc['Logistic Regression','Accuracy']:.2%}), MCC ({r.loc['Logistic Regression','MCC']:.3f}) and recall, "
       f"but LR has the higher 5-fold CV accuracy ({cv('Logistic Regression')} vs {cv('Support Vector Machine')}), the highest AUC, trains in milliseconds and is interpretable. "
       f"We use MCC and malignant sensitivity alongside accuracy because the malignant class is the safety-critical class; a malignant tumour predicted benign is the clinically costly error. "
       f"KNN misses more malignant cases than LR/SVM on this split, although it produces no benign false alarms. With only {len(y)} test rows, one sample changes accuracy by {1/len(y):.2%}, so the LR/SVM tie should not be over-interpreted.")
fig = lambda f, c: f"![{c}](artifacts/figures/{f}.png)\n*{c}*\n"
md = f"""# 🩺 Breast-Cancer Diagnosis – Five-Classifier Comparison & Streamlit Deployment
**BITS Pilani WILP · M.Tech (AIML) · Machine Learning · Assignment 2**

## a. Problem statement
Early, accurate discrimination between **malignant** and **benign** breast tumours from fine-needle-aspirate (FNA) cell-nucleus measurements reduces unnecessary biopsies and prevents missed cancers.
The task is a **binary classification** problem: build, evaluate and compare five classifiers (Logistic Regression, Decision Tree, KNN, Naive Bayes, SVM) on one dataset using six metrics
(Accuracy, AUC, Precision, Recall, F1, MCC), with malignant (class 0) consistently treated as the positive clinical condition. The interactive Streamlit app evaluates the *saved* models on an uploaded test CSV without retraining.

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
├── app.py                 # root Streamlit launcher used for deployment
├── train.py               # root training launcher used by the assignment
├── build_readme.py        # root documentation launcher
├── full_dataset.csv       # rubric-required full-data mirror
├── src/                   # application, training, and README-generation code
├── artifacts/             # generated data, models, metrics, and figures
├── test_data.csv          # rubric-required hold-out test data mirror
├── model/                 # rubric-required saved-model mirror
├── problem_statement/     # original assignment PDF
├── submission/            # final PDF and BITS Lab screenshots
├── config/                # student, link, and submission metadata
└── requirements.txt
```
The root `app.py`, `train.py`, and `build_readme.py` files are thin compatibility launchers; implementation remains in `src/`. `artifacts/` is the generated source-of-truth output, while root `model/`, `test_data.csv`, and `full_dataset.csv` are refreshed compatibility copies required by the assignment layout.
**Live Streamlit app:** {app}

### How to run / reproduce
```bash
pip install -r requirements.txt
python train.py            # retrains all models and regenerates artifacts/
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

### Additional diagnostic metrics
The required six metrics are shown above. For clinical interpretation, malignant (class 0) is treated as the positive condition in this section. Malignant sensitivity measures cancers correctly identified, and malignant FNR measures cancers predicted benign. Benign specificity measures the benign cases correctly classified as benign.

| ML Model Name | Balanced Accuracy | Malignant Sensitivity | Benign Specificity | Malignant FNR |
|---|---|---|---|---|
""" + "\n".join(
    f"| {n} | {diagnostic[n]['Balanced Accuracy']:.4f} | {diagnostic[n]['Malignant Sensitivity']:.4f} | "
    f"{diagnostic[n]['Benign Specificity']:.4f} | {diagnostic[n]['Malignant FNR']:.4f} |"
    for n in r.index
) + f"""

### Clinically labelled confusion counts
| ML Model Name | Malignant → malignant | Malignant → benign (missed cancer) | Benign → malignant (false alarm) | Benign → benign |
|---|---:|---:|---:|---:|
""" + "\n".join(
    f"| {n} | {diagnostic[n]['Malignant correct']} | {diagnostic[n]['Malignant missed']} | "
    f"{diagnostic[n]['Benign flagged']} | {diagnostic[n]['Benign correct']} |"
    for n in r.index
) + f"""

{fig('06_metric_comparison','Figure 3 – Metric comparison across models')}
{fig('04_roc_curves','Figure 4 – ROC curves (test set)')}
{fig('05_confusion_matrices','Figure 5 – Confusion matrices (benign/malignant axes; malignant is the clinical positive class)')}
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
CSV upload (test data only) · model dropdown · six required metric cards · detailed per-class classification report · confusion matrix · specificity · balanced accuracy · ROC curve · adjustable decision threshold ·
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
(ROOT / "README.md").write_text(md, encoding="utf-8")
print("README words:", len(md.split()))
