# 🩺 Breast-Cancer Diagnosis – Five-Classifier Comparison & Streamlit Deployment
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
| **Features** | 30 numeric (>= 12 required): 10 nucleus measurements × {mean, standard error, worst} |
| **Target** | `diagnosis` – 0 = malignant (212, 37.3 %), 1 = benign (357, 62.7 %) |
| **Class balance** | Moderately imbalanced (1 : 1.68) → MCC and recall reported alongside accuracy |
| **Missing values** | None |
| **Pre-processing** | `StandardScaler` inside each sklearn `Pipeline` (fit on training data only – no leakage); the saved `.joblib` therefore contains the complete preprocessing |

The 10 base measurements: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension.
Many are strongly collinear (radius–perimeter–area r > 0.95), which matters for Naive Bayes (see observations).

![Figure 1 – Class distribution](artifacts/figures/01_class_distribution.png)
*Figure 1 – Class distribution*

![Figure 2 – Feature correlation heatmap: blocks of highly correlated size features](artifacts/figures/02_correlation_heatmap.png)
*Figure 2 – Feature correlation heatmap: blocks of highly correlated size features*


## c. GitHub Repository Link
https://github.com/2026aj05790-apeksha/ml-assignment2

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
├── reports/               # grader feedback and reports
└── requirements.txt
```
The root `app.py`, `train.py`, and `build_readme.py` files are thin compatibility launchers; implementation remains in `src/`. `artifacts/` is the generated source-of-truth output, while root `model/`, `test_data.csv`, and `full_dataset.csv` are refreshed compatibility copies required by the assignment layout.
**Live Streamlit app:** https://ml-assignment2-bxkybtx2fackxbtb4bh9vk.streamlit.app

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
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9825 | 0.9954 | 0.9762 | 0.9762 | 0.9762 | 0.9623 |
| Decision Tree | 0.9035 | 0.9373 | 0.8444 | 0.9048 | 0.8736 | 0.7969 |
| KNN | 0.9737 | 0.9884 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |
| Naive Bayes | 0.9298 | 0.9868 | 0.9048 | 0.9048 | 0.9048 | 0.8492 |
| Support Vector Machine | 0.9825 | 0.9950 | 0.9762 | 0.9762 | 0.9762 | 0.9623 |

### Generalisation check (5-fold stratified CV on training data vs train vs test)
| ML Model Name | Train Acc | 5-fold CV Acc | Test Acc |
|---|---|---|---|
| Logistic Regression | 0.9890 | 0.978 ± 0.010 | 0.9825 |
| Decision Tree | 0.9802 | 0.916 ± 0.019 | 0.9035 |
| KNN | 1.0000 | 0.963 ± 0.020 | 0.9737 |
| Naive Bayes | 0.9385 | 0.934 ± 0.029 | 0.9298 |
| Support Vector Machine | 0.9824 | 0.969 ± 0.015 | 0.9825 |

### Additional diagnostic metrics
The required six metrics are shown above. For clinical interpretation, malignant (class 0) is treated as the positive condition in this section. Malignant sensitivity measures cancers correctly identified, and malignant FNR measures cancers predicted benign. Benign specificity measures the benign cases correctly classified as benign.

| ML Model Name | Balanced Accuracy | Malignant Sensitivity | Benign Specificity | Malignant FNR |
|---|---|---|---|---|
| Logistic Regression | 0.9812 | 0.9861 | 0.9762 | 0.0139 |
| Decision Tree | 0.9038 | 0.9028 | 0.9048 | 0.0972 |
| KNN | 0.9643 | 1.0000 | 0.9286 | 0.0000 |
| Naive Bayes | 0.9246 | 0.9444 | 0.9048 | 0.0556 |
| Support Vector Machine | 0.9812 | 0.9861 | 0.9762 | 0.0139 |

### Clinically labelled confusion counts
| ML Model Name | Malignant → malignant | Malignant → benign (missed cancer) | Benign → malignant (false alarm) | Benign → benign |
|---|---:|---:|---:|---:|
| Logistic Regression | 71 | 1 | 1 | 41 |
| Decision Tree | 65 | 7 | 4 | 38 |
| KNN | 72 | 0 | 3 | 39 |
| Naive Bayes | 68 | 4 | 4 | 38 |
| Support Vector Machine | 71 | 1 | 1 | 41 |

![Figure 3 – Metric comparison across models](artifacts/figures/06_metric_comparison.png)
*Figure 3 – Metric comparison across models*

![Figure 4 – ROC curves (test set)](artifacts/figures/04_roc_curves.png)
*Figure 4 – ROC curves (test set)*

![Figure 5 – Confusion matrices (class 0 = malignant)](artifacts/figures/05_confusion_matrices.png)
*Figure 5 – Confusion matrices (class 0 = malignant)*

![Figure 6 – Train vs CV vs test accuracy](artifacts/figures/08_overfit_check.png)
*Figure 6 – Train vs CV vs test accuracy*

![Figure 7 – Feature importance (Decision Tree) and coefficients (Logistic Regression)](artifacts/figures/09_feature_importance.png)
*Figure 7 – Feature importance (Decision Tree) and coefficients (Logistic Regression)*


### Observations on model performance
| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Logistic Regression records 98.25% accuracy, AUC 0.9954, and MCC 0.962. It makes 1 missed-cancer error and 1 benign false alarm on the 114-row test set. Its train score (0.989) is close to the cross-validation result (0.978 ± 0.010), while the linear coefficients provide useful directionality for interpretation. |
| Decision Tree | The Decision Tree has the lowest test performance: 90.35% accuracy, AUC 0.937, and MCC 0.797. Its training accuracy (0.980) is much higher than its test accuracy, indicating variance and overfitting. It misses 7 malignant cases, the largest missed-cancer count, although its rule structure remains easy to explain. |
| KNN | KNN reaches 97.37% accuracy and the best malignant sensitivity (1.000) because it misses no malignant test cases. The trade-off is 3 benign cases flagged as malignant, which lowers benign specificity to 0.929. Standardisation is essential for distance calculations; its training score of 1.0000 is higher than its 5-fold CV score (0.963 ± 0.020), showing that memorisation does not transfer perfectly. |
| Naive Bayes | Naive Bayes obtains 92.98% accuracy and MCC 0.849, while its AUC is relatively strong at 0.9868. The gap suggests useful ranking but less reliable threshold decisions. Highly correlated radius, perimeter, and area variables conflict with the independence assumption, and the similar train (0.939) and test scores point more to bias than overfitting. |
| Support Vector Machine | The RBF SVM is effectively tied with Logistic Regression: accuracy 98.25%, AUC 0.9950, and MCC 0.962. Its CV accuracy (0.969 ± 0.015) is below Logistic Regression's, so the additional non-linear flexibility does not improve this split despite the strong test result. |
| **Overall, Winner for your dataset?** | **Logistic Regression** (with SVM as a statistical tie). LR and SVM have identical test accuracy (98.25%), MCC (0.962) and recall, but LR has the higher 5-fold CV accuracy (0.978 ± 0.010 vs 0.969 ± 0.015), the highest AUC, trains in milliseconds and is interpretable. We use MCC and malignant sensitivity alongside accuracy because the malignant class is the safety-critical class; a malignant tumour predicted benign is the clinically costly error. KNN avoids missed malignant cases here but raises more benign false alarms. With only 114 test rows, one sample changes accuracy by 0.88%, so the LR/SVM tie should not be over-interpreted. |

### Error analysis
15 of 114 test samples are misclassified by at least one model and **1** by all five – these are genuinely borderline tumours whose measurements overlap the other class.
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
