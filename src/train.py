"""Train classifiers and write reproducible artifacts for the assignment."""
from pathlib import Path
import json, warnings
import numpy as np, pandas as pd, joblib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, roc_curve)
warnings.filterwarnings("ignore")
SEED = 42
sns.set_theme(style="whitegrid", context="notebook")
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "artifacts" / "data"
MODEL_DIR = ROOT / "artifacts" / "models"
COMPAT_MODEL_DIR = ROOT / "model"
RESULTS_DIR = ROOT / "artifacts" / "results"
FIGURES_DIR = ROOT / "artifacts" / "figures"
for directory in (DATA_DIR, MODEL_DIR, COMPAT_MODEL_DIR, RESULTS_DIR, FIGURES_DIR):
    directory.mkdir(parents=True, exist_ok=True)

# ---------- data ----------
raw = load_breast_cancer(as_frame=True)
df = raw.frame.rename(columns={"target": "diagnosis"})   # 0 = malignant, 1 = benign
X, y = df.drop(columns="diagnosis"), df["diagnosis"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
test_frame = pd.concat([X_te, y_te], axis=1)
test_frame.to_csv(DATA_DIR / "test_data.csv", index=False)
test_frame.to_csv(ROOT / "test_data.csv", index=False)
df.to_csv(DATA_DIR / "full_dataset.csv", index=False)
df.to_csv(ROOT / "full_dataset.csv", index=False)

# ---------- models (scaling lives INSIDE the pipeline, so saved file = full preprocessing) ----------
def pipe(clf): return Pipeline([("scaler", StandardScaler()), ("clf", clf)])
models = {
    "Logistic Regression": pipe(LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)),
    "Decision Tree": pipe(DecisionTreeClassifier(max_depth=5, min_samples_leaf=3, random_state=SEED)),
    "KNN": pipe(KNeighborsClassifier(n_neighbors=7, weights="distance")),
    "Naive Bayes": pipe(GaussianNB()),
    "Support Vector Machine": pipe(SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=SEED)),
}
FILES = {
    n: (
        MODEL_DIR / (n.lower().replace(" ", "_") + ".joblib"),
        COMPAT_MODEL_DIR / (n.lower().replace(" ", "_") + ".joblib"),
    )
    for n in models
}

cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
rows, probas, cms = [], {}, {}
for name, m in models.items():
    cv_acc = cross_val_score(m, X_tr, y_tr, cv=cv, scoring="accuracy")
    m.fit(X_tr, y_tr)
    for path in FILES[name]:
        joblib.dump(m, path)
    y_clinical = 1 - y_te  # malignant is the positive class for published metrics
    p = m.predict_proba(X_te)[:, 0]
    yp = (p >= 0.5).astype(int)
    probas[name] = p; cms[name] = confusion_matrix(y_clinical, yp)
    rows.append({"ML Model Name": name, "Accuracy": accuracy_score(y_clinical, yp), "AUC": roc_auc_score(y_clinical, p),
        "Precision": precision_score(y_clinical, yp), "Recall": recall_score(y_clinical, yp), "F1": f1_score(y_clinical, yp),
        "MCC": matthews_corrcoef(y_clinical, yp), "CV Acc (5-fold, train)": cv_acc.mean(), "CV Std": cv_acc.std(),
        "Train Acc": accuracy_score(y_tr, m.predict(X_tr))})
res = pd.DataFrame(rows).round(4)
res.to_csv(RESULTS_DIR / "metrics.csv", index=False)
with open(RESULTS_DIR / "confusion_matrices.json", "w") as handle:
    json.dump({n: c.tolist() for n, c in cms.items()}, handle)
print(res.to_string(index=False))

# ---------- figures ----------
COL = dict(zip(models, sns.color_palette("deep", 5)))
def save(name): plt.tight_layout(); plt.savefig(FIGURES_DIR / f"{name}.png", dpi=150); plt.close()

plt.figure(figsize=(5, 4)); ax = sns.countplot(x=y.map({0: "Malignant", 1: "Benign"}), palette=["#c44e52", "#55a868"])
for c in ax.containers: ax.bar_label(c)
plt.title("Class distribution (569 samples)"); plt.xlabel(""); save("01_class_distribution")

plt.figure(figsize=(11, 9)); sns.heatmap(X.corr(), cmap="coolwarm", center=0, square=True, cbar_kws={"shrink": .7})
plt.title("Feature correlation heatmap (30 features)"); save("02_correlation_heatmap")

top = X.corrwith(y).abs().sort_values(ascending=False).head(6).index
g = sns.pairplot(df[list(top) + ["diagnosis"]], hue="diagnosis", palette=["#c44e52", "#55a868"], corner=True, plot_kws=dict(s=12, alpha=.6))
g.figure.suptitle("Top-6 target-correlated features", y=1.02); g.figure.savefig(FIGURES_DIR / "03_pairplot_top_features.png", dpi=120); plt.close("all")

plt.figure(figsize=(7, 6))
for n, p in probas.items():
    fpr, tpr, _ = roc_curve(1 - y_te, p); plt.plot(fpr, tpr, lw=2, color=COL[n], label=f"{n} (AUC={roc_auc_score(1 - y_te, p):.3f})")
plt.plot([0, 1], [0, 1], "k--", lw=1); plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC curves – all models (test set)"); plt.legend(loc="lower right"); save("04_roc_curves")

fig, axs = plt.subplots(1, 5, figsize=(21, 4))
for ax, (n, c) in zip(axs, cms.items()):
    sns.heatmap(c, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax, xticklabels=["Malig", "Benign"], yticklabels=["Malig", "Benign"])
    ax.set_title(n); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
save("05_confusion_matrices")

m = res.melt(id_vars="ML Model Name", value_vars=["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"])
plt.figure(figsize=(12, 5)); sns.barplot(data=m, x="variable", y="value", hue="ML Model Name", palette=COL)
plt.ylim(0.8, 1.02); plt.xlabel(""); plt.ylabel("Score"); plt.title("Metric comparison (test set)")
plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left"); save("06_metric_comparison")

plt.figure(figsize=(8, 4.5)); sns.heatmap(res.set_index("ML Model Name")[["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]],
    annot=True, fmt=".3f", cmap="YlGnBu", vmin=.85, vmax=1); plt.ylabel(""); plt.title("Evaluation-metric heatmap"); save("07_metrics_heatmap")

x = np.arange(len(res)); plt.figure(figsize=(9, 4.5))
plt.bar(x - .2, res["Train Acc"], .4, label="Train accuracy"); plt.bar(x + .2, res["Accuracy"], .4, label="Test accuracy")
plt.errorbar(x, res["CV Acc (5-fold, train)"], yerr=res["CV Std"], fmt="ko", capsize=4, label="5-fold CV (mean±std)")
plt.xticks(x, res["ML Model Name"], rotation=15); plt.ylim(.85, 1.02); plt.legend(loc="lower left"); plt.title("Generalisation check: train vs CV vs test"); save("08_overfit_check")

dt = models["Decision Tree"].named_steps["clf"]; lr = models["Logistic Regression"].named_steps["clf"]
fig, axs = plt.subplots(1, 2, figsize=(14, 6))
pd.Series(dt.feature_importances_, X.columns).nlargest(10)[::-1].plot.barh(ax=axs[0], color="#4c72b0"); axs[0].set_title("Decision Tree – top-10 feature importance")
pd.Series(lr.coef_[0], X.columns).reindex(pd.Series(lr.coef_[0], X.columns).abs().nlargest(10).index)[::-1].plot.barh(ax=axs[1], color="#dd8452"); axs[1].set_title("Logistic Regression – top-10 |coefficients|")
save("09_feature_importance")
print("done")
