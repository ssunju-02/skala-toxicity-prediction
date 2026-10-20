"""
STEP 2. 학습/검증 분리 - Stratified 5-Fold (label 비율 유지)
STEP 3. 베이스라인 모델 - Logistic Regression / RandomForest / LightGBM / XGBoost
  - 입력: ECFP + FCFP + PTFP (3,072) + descriptor 4개 = 3,076 features
  - 평가: 5-Fold OOF(Out-of-Fold) 기준 ROC-AUC, Accuracy, F1, Precision, Recall
  - 주의: label 1 = 독성 無, 0 = 독성 有
결과물
  - folds.csv              : 행별 fold 번호 (STEP 4에서도 동일 분할 재사용)
  - baseline_outputs/      : 성능표, OOF 예측값, ROC 곡선
"""
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

SEED = 42
N_SPLITS = 5
BASE = Path(__file__).resolve().parent
OUT = BASE / "baseline_outputs"
OUT.mkdir(exist_ok=True)

DESC_COLS = ["MolWt", "clogp", "sa_score", "qed"]

# ------------------------------------------------------------------
# 데이터 로드
# ------------------------------------------------------------------
df = pd.read_csv(BASE / "train.csv")
fp_cols = [c for c in df.columns if c.startswith(("ecfp_", "fcfp_", "ptfp_"))]
feat_cols = fp_cols + DESC_COLS
X = df[feat_cols].to_numpy(dtype=np.float32)
y = df["label"].to_numpy()
print(f"X: {X.shape}, y 분포: {np.bincount(y)} (0=독성有, 1=독성無)")

# ------------------------------------------------------------------
# STEP 2. Stratified K-Fold
# ------------------------------------------------------------------
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
folds = np.full(len(df), -1)
for k, (_, va_idx) in enumerate(skf.split(X, y)):
    folds[va_idx] = k
pd.DataFrame({"SMILES": df["SMILES"], "label": y, "fold": folds}).to_csv(BASE / "folds.csv", index=False)

print("\n[STEP 2] fold별 크기 및 label=1(독성無) 비율")
for k in range(N_SPLITS):
    m = folds == k
    print(f"  fold {k}: valid {m.sum():>4}건 | train {(~m).sum():>4}건 | "
          f"valid 독성無 비율 {y[m].mean()*100:.2f}% | train 독성無 비율 {y[~m].mean()*100:.2f}%")

# ------------------------------------------------------------------
# STEP 3. 베이스라인 모델
# ------------------------------------------------------------------
models = {
    "LogisticRegression": lambda: make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.01, max_iter=3000, random_state=SEED)),
    "RandomForest": lambda: RandomForestClassifier(
        n_estimators=500, max_features="sqrt", n_jobs=-1, random_state=SEED),
    "LightGBM": lambda: LGBMClassifier(
        n_estimators=500, learning_rate=0.05, num_leaves=31, colsample_bytree=0.3,
        subsample=0.8, subsample_freq=1, random_state=SEED, n_jobs=-1, verbose=-1),
    "XGBoost": lambda: XGBClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=6, colsample_bytree=0.3,
        subsample=0.8, tree_method="hist", eval_metric="logloss",
        random_state=SEED, n_jobs=-1),
}


def metrics(y_true, prob, thr=0.5):
    pred = (prob >= thr).astype(int)
    return {
        "ROC-AUC": roc_auc_score(y_true, prob),
        "Accuracy": accuracy_score(y_true, pred),
        "F1": f1_score(y_true, pred),
        "Precision": precision_score(y_true, pred),
        "Recall": recall_score(y_true, pred),
    }


results, fold_aucs = [], {}
oof = pd.DataFrame({"SMILES": df["SMILES"], "label": y, "fold": folds})

for name, make_model in models.items():
    print(f"\n[STEP 3] {name} 학습 중...")
    t0 = time.time()
    prob = np.zeros(len(y))
    aucs = []
    for k in range(N_SPLITS):
        tr, va = folds != k, folds == k
        model = make_model()
        model.fit(X[tr], y[tr])
        prob[va] = model.predict_proba(X[va])[:, 1]
        aucs.append(roc_auc_score(y[va], prob[va]))
        print(f"   fold {k}: AUC {aucs[-1]:.4f}")
    oof[name] = prob
    fold_aucs[name] = aucs
    m = metrics(y, prob)
    m.update({"Model": name, "AUC_fold_mean": np.mean(aucs), "AUC_fold_std": np.std(aucs),
              "Time(s)": time.time() - t0})
    results.append(m)
    print(f"   → OOF AUC {m['ROC-AUC']:.4f} | fold 평균 {np.mean(aucs):.4f} ± {np.std(aucs):.4f} "
          f"| Acc {m['Accuracy']:.4f} | F1 {m['F1']:.4f} | {m['Time(s)']:.1f}s")

res = (pd.DataFrame(results)
       .set_index("Model")[["ROC-AUC", "AUC_fold_mean", "AUC_fold_std", "Accuracy",
                            "F1", "Precision", "Recall", "Time(s)"]]
       .sort_values("ROC-AUC", ascending=False))

# 참고: 4개 트리/선형 모델 단순 평균 앙상블
avg = oof[list(models)].mean(axis=1).to_numpy()
avg_m = metrics(y, avg)
print("\n" + "=" * 80)
print("  베이스라인 성능 요약 (5-Fold OOF, threshold=0.5)")
print("=" * 80)
print(res.round(4).to_string())
print(f"\n  (참고) 4개 모델 단순평균 앙상블 OOF AUC {avg_m['ROC-AUC']:.4f} | "
      f"Acc {avg_m['Accuracy']:.4f} | F1 {avg_m['F1']:.4f}")

res.round(4).to_csv(OUT / "baseline_results.csv")
oof.to_csv(OUT / "oof_predictions.csv", index=False)

# ROC 곡선
fig, ax = plt.subplots(figsize=(6, 6))
for name in res.index:
    fpr, tpr, _ = roc_curve(y, oof[name])
    ax.plot(fpr, tpr, label=f"{name} (AUC={res.loc[name, 'ROC-AUC']:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("베이스라인 ROC (5-Fold OOF)")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(OUT / "roc_baseline.png", dpi=120)
plt.close(fig)

# fold별 AUC 박스플롯
fig, ax = plt.subplots(figsize=(7, 4))
ax.boxplot([fold_aucs[n] for n in res.index], labels=res.index)
ax.set_title("fold별 ROC-AUC 분포")
ax.set_ylabel("ROC-AUC")
fig.tight_layout()
fig.savefig(OUT / "fold_auc_box.png", dpi=120)
plt.close(fig)

print(f"\n저장 완료: {BASE / 'folds.csv'}, {OUT}")
