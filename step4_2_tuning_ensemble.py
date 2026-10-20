"""
STEP 4-2. 하이퍼파라미터 튜닝(Optuna) + 앙상블 + 임계값(threshold) 최적화
  - feature set: step4_outputs/best_features.json (STEP 4-1 결과)
  - 튜닝 대상: LightGBM, XGBoost (목표: 5-Fold OOF ROC-AUC 최대화)
  - 앙상블: 튜닝된 LGBM(seed 3개 평균) + XGB + RandomForest → 가중 평균
  - 평가지표: 발표자료 기준 F1 (Positive = 1 = 독성 無) → OOF로 F1 최대 threshold 선택
결과물: step4_outputs/final_config.json, tuning_results.csv, ensemble_results.csv, oof_final.csv
"""
import itertools
import json
import time
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

import features as F

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

SEED = 42
N_TRIALS_LGBM = 30
N_TRIALS_XGB = 20
BASE = Path(__file__).resolve().parent
OUT = BASE / "step4_outputs"

df = pd.read_csv(BASE / "train.csv")
folds = pd.read_csv(BASE / "folds.csv")["fold"].to_numpy()
y = df["label"].to_numpy()
K = folds.max() + 1
X_all, G = F.build_features(df, "train")
best_feat = json.load(open(OUT / "best_features.json"))
cols = F.select_cols(G, best_feat["groups"])
print(f"feature set: {best_feat['experiment']} / selection={best_feat['selection']}")

# fold별 선택 컬럼 미리 계산 (train 데이터로만 선택 → 누수 없음)
fold_cols = []
for k in range(K):
    tr = folds != k
    fold_cols.append(F.apply_selection(X_all.loc[tr], y[tr], cols, best_feat["selection"]))
print(f"fold별 사용 feature 수: {[len(c) for c in fold_cols]}")

# numpy 캐시 (속도)
fold_data = []
for k in range(K):
    tr, va = folds != k, folds == k
    Xk = X_all[fold_cols[k]].to_numpy(np.float32)
    fold_data.append((Xk[tr], y[tr], Xk[va], va))


def cv_oof(make_model):
    oof = np.zeros(len(y))
    for X_tr, y_tr, X_va, va in fold_data:
        oof[va] = make_model().fit(X_tr, y_tr).predict_proba(X_va)[:, 1]
    return oof


# ------------------------------------------------------------------
# 1) LightGBM 튜닝
# ------------------------------------------------------------------
def lgbm_params(t):
    return dict(
        n_estimators=t.suggest_int("n_estimators", 300, 1500, step=100),
        learning_rate=t.suggest_float("learning_rate", 0.01, 0.1, log=True),
        num_leaves=t.suggest_int("num_leaves", 15, 127),
        max_depth=t.suggest_int("max_depth", -1, 12),
        min_child_samples=t.suggest_int("min_child_samples", 5, 60),
        colsample_bytree=t.suggest_float("colsample_bytree", 0.05, 0.6),
        subsample=t.suggest_float("subsample", 0.5, 1.0),
        reg_alpha=t.suggest_float("reg_alpha", 1e-3, 10, log=True),
        reg_lambda=t.suggest_float("reg_lambda", 1e-3, 10, log=True),
    )


def make_lgbm(p, seed=SEED):
    return lambda: LGBMClassifier(**p, subsample_freq=1, random_state=seed, n_jobs=-1, verbose=-1)


def make_xgb(p, seed=SEED):
    return lambda: XGBClassifier(**p, tree_method="hist", eval_metric="logloss",
                                 random_state=seed, n_jobs=-1)


def tune(name, param_fn, maker, n_trials, baseline):
    t0 = time.time()
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.enqueue_trial(baseline)  # 베이스라인 파라미터를 첫 trial로 → 최소 베이스라인 이상 보장

    def objective(t):
        auc = roc_auc_score(y, cv_oof(maker(param_fn(t))))
        print(f"   {name} trial {t.number:>2}: AUC {auc:.4f} (best {max(auc, study.best_value if t.number else auc):.4f})")
        return auc

    study.optimize(objective, n_trials=n_trials)
    print(f"  → {name} best AUC {study.best_value:.4f} ({time.time()-t0:.0f}s)\n    {study.best_params}")
    return study


print(f"\n[1] LightGBM 튜닝 ({N_TRIALS_LGBM} trials)")
lgbm_base = dict(n_estimators=500, learning_rate=0.05, num_leaves=31, max_depth=-1,
                 min_child_samples=20, colsample_bytree=0.3, subsample=0.8,
                 reg_alpha=1e-3, reg_lambda=1e-3)
st_lgbm = tune("LGBM", lgbm_params, make_lgbm, N_TRIALS_LGBM, lgbm_base)


# ------------------------------------------------------------------
# 2) XGBoost 튜닝
# ------------------------------------------------------------------
def xgb_params(t):
    return dict(
        n_estimators=t.suggest_int("n_estimators", 300, 1500, step=100),
        learning_rate=t.suggest_float("learning_rate", 0.01, 0.1, log=True),
        max_depth=t.suggest_int("max_depth", 3, 10),
        min_child_weight=t.suggest_float("min_child_weight", 0.5, 10, log=True),
        colsample_bytree=t.suggest_float("colsample_bytree", 0.05, 0.6),
        subsample=t.suggest_float("subsample", 0.5, 1.0),
        gamma=t.suggest_float("gamma", 1e-3, 2, log=True),
        reg_lambda=t.suggest_float("reg_lambda", 1e-2, 10, log=True),
    )


print(f"\n[2] XGBoost 튜닝 ({N_TRIALS_XGB} trials)")
xgb_base = dict(n_estimators=500, learning_rate=0.05, max_depth=6, min_child_weight=1,
                colsample_bytree=0.3, subsample=0.8, gamma=1e-3, reg_lambda=1)
st_xgb = tune("XGB", xgb_params, make_xgb, N_TRIALS_XGB, xgb_base)

pd.concat([st_lgbm.trials_dataframe().assign(model="LightGBM"),
           st_xgb.trials_dataframe().assign(model="XGBoost")]).to_csv(OUT / "tuning_results.csv", index=False)

# ------------------------------------------------------------------
# 3) 최종 OOF: LGBM seed 평균 / XGB / RF
# ------------------------------------------------------------------
print("\n[3] 최종 모델 OOF 생성")
SEEDS = [42, 7, 2024]
oof = {}
oof["LGBM"] = np.mean([cv_oof(make_lgbm(st_lgbm.best_params, s)) for s in SEEDS], axis=0)
oof["XGB"] = np.mean([cv_oof(make_xgb(st_xgb.best_params, s)) for s in SEEDS], axis=0)
rf_params = dict(n_estimators=800, max_features="sqrt", min_samples_leaf=1)
oof["RF"] = cv_oof(lambda: RandomForestClassifier(**rf_params, n_jobs=-1, random_state=SEED))
for n, p in oof.items():
    print(f"  {n:<5} OOF AUC {roc_auc_score(y, p):.4f}")

# ------------------------------------------------------------------
# 4) 앙상블 가중치 탐색 (0.1 단위 그리드)
# ------------------------------------------------------------------
print("\n[4] 앙상블")
names = list(oof)
P = np.column_stack([oof[n] for n in names])
ens_rows = []
for w in itertools.product(np.arange(0, 1.01, 0.1), repeat=len(names)):
    if abs(sum(w) - 1) > 1e-6:
        continue
    auc = roc_auc_score(y, P @ np.array(w))
    ens_rows.append({**{f"w_{n}": round(x, 1) for n, x in zip(names, w)}, "OOF_AUC": auc})
ens = pd.DataFrame(ens_rows).sort_values("OOF_AUC", ascending=False)
ens.round(4).to_csv(OUT / "ensemble_results.csv", index=False)
print(ens.head(10).round(4).to_string(index=False))
best_w = np.array([ens.iloc[0][f"w_{n}"] for n in names])
final_prob = P @ best_w


# ------------------------------------------------------------------
# 5) Threshold 최적화 (F1 기준) + 성능 요약
# ------------------------------------------------------------------
def summarize(prob, thr):
    pred = (prob >= thr).astype(int)
    return {"AUC": roc_auc_score(y, prob), "F1": f1_score(y, pred), "Accuracy": accuracy_score(y, pred)}


ths = np.arange(0.2, 0.81, 0.01)
f1s = [f1_score(y, (final_prob >= t).astype(int)) for t in ths]
accs = [accuracy_score(y, (final_prob >= t).astype(int)) for t in ths]
thr_f1 = float(ths[int(np.argmax(f1s))])
thr_acc = float(ths[int(np.argmax(accs))])

print("\n" + "=" * 80)
print("  STEP 4 최종 요약 (5-Fold OOF)")
print("=" * 80)
summary = pd.DataFrame({
    "STEP3 베이스라인 LGBM (thr 0.5)": summarize(
        pd.read_csv(BASE / "baseline_outputs" / "oof_predictions.csv")["LightGBM"].to_numpy(), 0.5),
    "튜닝 LGBM (thr 0.5)": summarize(oof["LGBM"], 0.5),
    "튜닝 XGB (thr 0.5)": summarize(oof["XGB"], 0.5),
    "앙상블 (thr 0.5)": summarize(final_prob, 0.5),
    f"앙상블 (F1 최적 thr {thr_f1:.2f})": summarize(final_prob, thr_f1),
    f"앙상블 (Acc 최적 thr {thr_acc:.2f})": summarize(final_prob, thr_acc),
}).T
print(summary.round(4).to_string())

config = {
    "feature": best_feat,
    "lgbm_params": st_lgbm.best_params, "lgbm_seeds": SEEDS,
    "xgb_params": st_xgb.best_params, "xgb_seeds": SEEDS,
    "rf_params": rf_params,
    "weights": dict(zip(names, best_w.round(2).tolist())),
    "threshold_f1": thr_f1, "threshold_acc": thr_acc,
    "oof": {k: {m: round(float(v), 4) for m, v in r.items()} for k, r in summary.iterrows()},
}
json.dump(config, open(OUT / "final_config.json", "w"), ensure_ascii=False, indent=2)
pd.DataFrame({"SMILES": df["SMILES"], "label": y, "fold": folds, **oof, "ensemble": final_prob}) \
    .to_csv(OUT / "oof_final.csv", index=False)
print(f"\n저장: {OUT / 'final_config.json'}")
