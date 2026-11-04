"""Streamlit app for evaluating the saved breast-cancer classifiers."""
from pathlib import Path
import os, joblib
import numpy as np, pandas as pd, streamlit as st
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, confusion_matrix, classification_report, roc_curve)

st.set_page_config(page_title="Tumour Diagnosis – Model Lab", page_icon="🩺", layout="wide")
ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "artifacts" / "models"
TEST_DATA = ROOT / "artifacts" / "data" / "test_data.csv"
MODELS = {"Logistic Regression": "logistic_regression", "Decision Tree": "decision_tree", "KNN": "knn",
          "Naive Bayes": "naive_bayes", "Support Vector Machine": "support_vector_machine"}
TARGET, CLASSES = "diagnosis", ["Malignant (0)", "Benign (1)"]

@st.cache_resource
def load(name): return joblib.load(MODEL_DIR / (MODELS[name] + ".joblib"))

def metrics(y, yp, p):
    return {"Accuracy": accuracy_score(y, yp), "AUC": roc_auc_score(y, p), "Precision": precision_score(y, yp),
            "Recall": recall_score(y, yp), "F1": f1_score(y, yp), "MCC": matthews_corrcoef(y, yp)}

# ---------------- sidebar ----------------
st.sidebar.title("⚙️ Controls")
st.sidebar.caption("BITS Pilani · M.Tech AIML · ML Assignment 2")
up = st.sidebar.file_uploader("1 · Upload TEST data (CSV)", type="csv",
        help="Must contain all 30 feature columns and the `diagnosis` target column (0/1).")
use_demo = st.sidebar.checkbox("Use bundled test_data.csv", value=up is None, disabled=up is None)
model_name = st.sidebar.selectbox("2 · Choose model", list(MODELS))
thr = st.sidebar.slider("3 · Decision threshold (P(benign))", 0.05, 0.95, 0.50, 0.05)
with open(TEST_DATA, "rb") as f:
    st.sidebar.download_button("⬇️ Download sample test_data.csv", f, "test_data.csv", "text/csv")

st.title("🩺 Breast-Cancer Diagnosis – Model Evaluation Lab")
st.markdown("Evaluate **five pre-trained classifiers** on any uploaded test CSV. "
            "Models are loaded from disk – **no retraining** happens here.")

df = pd.read_csv(up) if up is not None else pd.read_csv(TEST_DATA)
model = load(model_name)
feats = list(model.feature_names_in_)
missing = [c for c in feats + [TARGET] if c not in df.columns]
if missing:
    st.error(f"Uploaded CSV is missing required columns: {missing}"); st.stop()
target = pd.to_numeric(df[TARGET], errors="coerce")
if target.isna().any() or not target.isin([0, 1]).all():
    st.error("Target column must contain only 0 / 1."); st.stop()
non_numeric = [c for c in feats if not pd.api.types.is_numeric_dtype(df[c])]
if non_numeric:
    st.error(f"Feature columns must be numeric: {non_numeric}"); st.stop()
before = len(df)
df = df.dropna(subset=feats + [TARGET]).copy()
if len(df) < before:
    st.warning(f"Dropped {before - len(df)} row(s) containing missing values.")
if df.empty:
    st.error("No valid rows remain after removing missing values."); st.stop()
df[TARGET] = target.loc[df.index].astype(int)
if df[TARGET].nunique() < 2:
    st.warning("At least two target classes are required to compute AUC and MCC."); st.stop()
X, y = df[feats], df[TARGET]
p = model.predict_proba(X)[:, 1]; yp = (p >= thr).astype(int)
M = metrics(y, yp, p)

tab1, tab2, tab3, tab4 = st.tabs(["📊 Metrics", "🧮 Confusion matrix & report", "🆚 Compare all models", "🔍 Data & predictions"])

with tab1:
    st.subheader(f"{model_name} – {len(df)} test rows")
    for col, (k, v) in zip(st.columns(6), M.items()): col.metric(k, f"{v:.4f}")
    c1, c2 = st.columns(2)
    fig, ax = plt.subplots(figsize=(5, 4)); fpr, tpr, _ = roc_curve(y, p)
    ax.plot(fpr, tpr, lw=2, label=f"AUC={M['AUC']:.3f}"); ax.plot([0, 1], [0, 1], "k--")
    ax.set(xlabel="FPR", ylabel="TPR", title="ROC curve"); ax.legend(); c1.pyplot(fig)
    fig, ax = plt.subplots(figsize=(5, 4)); ax.bar(M.keys(), M.values(), color=sns.color_palette("viridis", 6))
    ax.set_ylim(0, 1.05); ax.set_title("Metric profile"); plt.setp(ax.get_xticklabels(), rotation=30); c2.pyplot(fig)

with tab2:
    cm = confusion_matrix(y, yp, labels=[0, 1]); c1, c2 = st.columns([1, 1.2])
    fig, ax = plt.subplots(figsize=(4.5, 4)); sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=CLASSES, yticklabels=CLASSES); ax.set(xlabel="Predicted", ylabel="Actual", title="Confusion matrix"); c1.pyplot(fig)
    c2.markdown("**Classification report**")
    c2.dataframe(pd.DataFrame(classification_report(y, yp, target_names=CLASSES, output_dict=True)).T.round(3), width="stretch")
    tn, fp, fn, tp = cm.ravel()
    st.info(f"**{fn}** malignant case(s) missed (false negatives – clinically the costliest error) and **{fp}** benign case(s) flagged as malignant.")

with tab3:
    rows = []
    for n in MODELS:
        m = load(n); pp = m.predict_proba(X)[:, 1]; rows.append({"Model": n, **metrics(y, (pp >= .5).astype(int), pp)})
    cmp_df = pd.DataFrame(rows).set_index("Model")
    st.dataframe(cmp_df.style.highlight_max(axis=0, color="#b7e4c7").format("{:.4f}"), width="stretch")
    st.success(f"🏆 Best MCC on this data: **{cmp_df['MCC'].idxmax()}** ({cmp_df['MCC'].max():.4f})  (compared at threshold 0.5)")
    fig, ax = plt.subplots(figsize=(9, 4)); cmp_df.plot.bar(ax=ax, rot=15, colormap="tab10"); ax.set_ylim(.7, 1.02)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left"); st.pyplot(fig)

with tab4:
    out = df.copy(); out["P(benign)"] = p.round(4); out["Predicted"] = yp; out["Correct"] = out["Predicted"] == out[TARGET]
    st.dataframe(out, width="stretch", height=350)
    st.download_button("⬇️ Download predictions", out.to_csv(index=False).encode(), "predictions.csv", "text/csv")
    st.caption(f"Shape: {df.shape} · class balance: {y.value_counts().to_dict()}")
