"""
STEP 4-1. Feature 조합 실험 + RDKit descriptor 추가 + Feature Selection
  - 모델은 STEP 3 최고 성능인 LightGBM(베이스라인 파라미터)으로 고정
  - 분할은 folds.csv (STEP 2와 동일한 5-Fold) 재사용 → 실험 간 공정 비교
  - 평가지표: 5-Fold OOF ROC-AUC
결과물: step4_outputs/feature_experiments.csv, feature_experiments.png, best_features.json
"""
import json
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

import features as F

warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

SEED = 42
BASE = Path(__file__).resolve().parent
OUT = BASE / "step4_outputs"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(BASE / "train.csv")
folds = pd.read_csv(BASE / "folds.csv")["fold"].to_numpy()
y = df["label"].to_numpy()
X_all, G = F.build_features(df, "train")
K = folds.max() + 1


def lgbm():
    return LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=31,
                          colsample_bytree=0.3, subsample=0.8, subsample_freq=1,
                          random_state=SEED, n_jobs=-1, verbose=-1)


def run_cv(cols, select_fn=None, fold_ids=folds):
    """5-Fold OOF AUC. select_fn(X_tr, y_tr, cols) → fold 내부(train만 사용)에서 컬럼 선택"""
    oof = np.zeros(len(y))
    n_used = []
    for k in range(fold_ids.max() + 1):
        tr, va = fold_ids != k, fold_ids == k
        use = cols if select_fn is None else select_fn(X_all.loc[tr, cols], y[tr], cols)
        n_used.append(len(use))
        m = lgbm().fit(X_all.loc[tr, use], y[tr])
        oof[va] = m.predict_proba(X_all.loc[va, use])[:, 1]
    fold_auc = [roc_auc_score(y[fold_ids == k], oof[fold_ids == k]) for k in range(fold_ids.max() + 1)]
    return roc_auc_score(y, oof), np.std(fold_auc), int(np.mean(n_used)), oof


results = []


def record(group, name, cols, select_fn=None):
    t0 = time.time()
    auc, std, n, oof = run_cv(cols, select_fn)
    results.append({"group": group, "experiment": name, "n_features": n,
                    "OOF_AUC": auc, "fold_std": std, "time(s)": time.time() - t0})
    print(f"  [{group}] {name:<38} feat {n:>5} | AUC {auc:.4f} ± {std:.4f} | {time.time()-t0:5.1f}s")
    return auc, oof


# ------------------------------------------------------------------
# (A) Feature 조합 실험
# ------------------------------------------------------------------
print("\n(A) Feature 조합 실험 (LightGBM)")
combos = {
    "ECFP만": ["ecfp"],
    "FCFP만": ["fcfp"],
    "PTFP만": ["ptfp"],
    "descriptor 4개만": ["desc"],
    "ECFP+FCFP+PTFP": ["ecfp", "fcfp", "ptfp"],
    "FP 전체+descriptor (=베이스라인)": ["ecfp", "fcfp", "ptfp", "desc"],
}
for name, g in combos.items():
    record("A.조합", name, F.select_cols(G, g))

# ------------------------------------------------------------------
# (B) RDKit descriptor 추가
# ------------------------------------------------------------------
print("\n(B) RDKit feature 추가")
combos_b = {
    "RDKit descriptor만": ["rdkit"],
    "RDKit descriptor + desc4": ["rdkit", "desc"],
    "베이스라인 + RDKit": ["ecfp", "fcfp", "ptfp", "desc", "rdkit"],
    "베이스라인 + RDKit + MACCS": ["ecfp", "fcfp", "ptfp", "desc", "rdkit", "maccs"],
}
for name, g in combos_b.items():
    record("B.RDKit", name, F.select_cols(G, g))

# 지금까지 최고 조합 선택
res_df = pd.DataFrame(results)
best_row = res_df.loc[res_df["OOF_AUC"].idxmax()]
best_groups = {**combos, **combos_b}[best_row["experiment"]]
best_cols = F.select_cols(G, best_groups)
print(f"\n  → 최고 조합: {best_row['experiment']} (AUC {best_row['OOF_AUC']:.4f})")

# ------------------------------------------------------------------
# (C) Feature Selection (최고 조합 기준, 반드시 fold의 train 데이터로만 선택 → 누수 방지)
# ------------------------------------------------------------------
print("\n(C) Feature Selection")


def drop_rare_bits(X_tr, y_tr, cols, lo=0.01, hi=0.99):
    """등장 빈도 1% 미만 / 99% 초과인 0-1 bit 제거 (연속형 변수는 유지)"""
    keep = []
    for c in cols:
        if c.startswith(("ecfp_", "fcfp_", "ptfp_", "maccs_")):
            p = X_tr[c].mean()
            if lo <= p <= hi:
                keep.append(c)
        elif X_tr[c].nunique(dropna=True) > 1:
            keep.append(c)
    return keep


def top_k_importance(k):
    def fn(X_tr, y_tr, cols):
        m = LGBMClassifier(n_estimators=300, learning_rate=0.05, colsample_bytree=0.3,
                           importance_type="gain", random_state=SEED, n_jobs=-1, verbose=-1)
        m.fit(X_tr, y_tr)
        imp = pd.Series(m.feature_importances_, index=cols)
        return imp.sort_values(ascending=False).index[:k].tolist()
    return fn


record("C.Selection", "희귀/상수 bit 제거 (1%~99%)", best_cols, drop_rare_bits)
for k in [300, 500, 1000, 1500]:
    record("C.Selection", f"Importance 상위 {k}개", best_cols, top_k_importance(k))

# ------------------------------------------------------------------
# (D) 검증 방식 점검: FP가 동일한 분자를 같은 fold로 묶었을 때 (StratifiedGroupKFold)
# ------------------------------------------------------------------
print("\n(D) 검증 신뢰도 점검 - FP 동일 분자 그룹핑")
fp_cols = F.select_cols(G, ["ecfp", "fcfp", "ptfp"])
grp = pd.util.hash_pandas_object(df[fp_cols], index=False).to_numpy()
sgkf = StratifiedGroupKFold(n_splits=K, shuffle=True, random_state=SEED)
gfolds = np.full(len(y), -1)
for k, (_, va) in enumerate(sgkf.split(X_all, y, grp)):
    gfolds[va] = k
auc_s, _, _, _ = run_cv(best_cols, fold_ids=folds)
auc_g, std_g, _, _ = run_cv(best_cols, fold_ids=gfolds)
print(f"  StratifiedKFold AUC {auc_s:.4f} vs StratifiedGroupKFold AUC {auc_g:.4f} "
      f"(차이 {auc_s - auc_g:+.4f})")

# ------------------------------------------------------------------
# 결과 저장
# ------------------------------------------------------------------
res_df = pd.DataFrame(results)
res_df.round(4).to_csv(OUT / "feature_experiments.csv", index=False)
print("\n" + "=" * 90)
print(res_df.round(4).to_string(index=False))

# 최종 feature set 결정: 최고 AUC (동률 수준이면 feature 수가 적은 쪽)
best = res_df.sort_values("OOF_AUC", ascending=False).iloc[0]
print(f"\n★ 최종 선택: [{best['group']}] {best['experiment']} | AUC {best['OOF_AUC']:.4f} | feat {best['n_features']}")
selection = None
if best["group"] == "C.Selection":
    selection = ("rare" if "희귀" in best["experiment"]
                 else int(best["experiment"].split()[2].replace("개", "")))
json.dump({"groups": best_groups, "selection": selection,
           "oof_auc": float(best["OOF_AUC"]), "experiment": best["experiment"],
           "group_kfold_auc": float(auc_g)},
          open(OUT / "best_features.json", "w"), ensure_ascii=False, indent=2)

fig, ax = plt.subplots(figsize=(9, 7))
order = res_df.sort_values("OOF_AUC")
colors = order["group"].map({"A.조합": "#7f8fa6", "B.RDKit": "#44bd32", "C.Selection": "#e1b12c"})
ax.barh(order["experiment"], order["OOF_AUC"], xerr=order["fold_std"], color=colors)
ax.axvline(res_df.loc[res_df.experiment.str.contains("베이스라인\\)"), "OOF_AUC"].iloc[0],
           ls="--", c="red", label="STEP 3 베이스라인")
ax.set_xlim(order["OOF_AUC"].min() - 0.03, order["OOF_AUC"].max() + 0.02)
ax.set_xlabel("5-Fold OOF ROC-AUC")
ax.set_title("Feature 실험 결과 (LightGBM)")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "feature_experiments.png", dpi=120)
print(f"\n저장: {OUT}")
