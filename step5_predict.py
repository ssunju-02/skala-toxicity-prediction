"""
STEP 5. 예측 & 제출 파일 생성
  - STEP 4 최종 설정(step4_outputs/final_config.json)으로 train.csv 전체(8,349건)를 다시 학습
  - predict_input.csv(927건) 예측 → predict_output_김선주.csv (SMILES, output) 저장
  - output: 1 = 독성 無, 0 = 독성 有 / threshold는 OOF 기준 F1 최적값 사용
"""
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import features as F

warnings.filterwarnings("ignore")

AUTHOR = "김선주"
BASE = Path(__file__).resolve().parent
INPUT = Path(os.environ.get("PREDICT_INPUT", BASE / "predict_input.csv"))
OUTPUT = Path(os.environ.get("PREDICT_OUTPUT", BASE / f"predict_output_{AUTHOR}.csv"))
cfg = json.load(open(BASE / "step4_outputs" / "final_config.json"))
THRESHOLD = cfg["threshold_f1"]

if not INPUT.exists():
    sys.exit(f"[오류] {INPUT} 파일이 없습니다. MiniProject 폴더에 predict_input.csv를 넣어 주세요.")

# ------------------------------------------------------------------
# 1) 데이터 로드 & 구조 점검
# ------------------------------------------------------------------
train = pd.read_csv(BASE / "train.csv")
test = pd.read_csv(INPUT)
print(f"train {train.shape}, predict_input {test.shape}")

expected = [c for c in train.columns if c != "label"]
missing = set(expected) - set(test.columns)
assert not missing, f"predict_input에 없는 컬럼: {sorted(missing)[:10]}"
print(f"  컬럼 구조 일치 ✔ / 결측 셀 {test[expected].isna().sum().sum()} / "
      f"중복 SMILES {test['SMILES'].duplicated().sum()} / train과 겹치는 SMILES {test['SMILES'].isin(train['SMILES']).sum()}")

# ------------------------------------------------------------------
# 2) Feature 생성 (train과 동일한 함수/컬럼 순서)
# ------------------------------------------------------------------
X_tr_all, G = F.build_features(train, "train")
X_te_all, _ = F.build_features(test[expected], INPUT.stem)
cols = F.select_cols(G, cfg["feature"]["groups"])
cols = F.apply_selection(X_tr_all, train["label"].to_numpy(), cols, cfg["feature"]["selection"])
X_tr = X_tr_all[cols].to_numpy(np.float32)
X_te = X_te_all[cols].to_numpy(np.float32)
y = train["label"].to_numpy()
print(f"  사용 feature 수: {len(cols)}")

# ------------------------------------------------------------------
# 3) 전체 train으로 최종 모델 학습 → 앙상블 예측
# ------------------------------------------------------------------
w = cfg["weights"]
probs = {}
if w.get("LGBM", 0) > 0:
    probs["LGBM"] = np.mean([
        LGBMClassifier(**cfg["lgbm_params"], subsample_freq=1, random_state=s, n_jobs=-1, verbose=-1)
        .fit(X_tr, y).predict_proba(X_te)[:, 1] for s in cfg["lgbm_seeds"]], axis=0)
    print("  LightGBM 학습/예측 완료")
if w.get("XGB", 0) > 0:
    probs["XGB"] = np.mean([
        XGBClassifier(**cfg["xgb_params"], tree_method="hist", eval_metric="logloss",
                      random_state=s, n_jobs=-1)
        .fit(X_tr, y).predict_proba(X_te)[:, 1] for s in cfg["xgb_seeds"]], axis=0)
    print("  XGBoost 학습/예측 완료")
if w.get("RF", 0) > 0:
    probs["RF"] = RandomForestClassifier(**cfg["rf_params"], n_jobs=-1, random_state=42) \
        .fit(X_tr, y).predict_proba(X_te)[:, 1]
    print("  RandomForest 학습/예측 완료")

prob = sum(w[n] * p for n, p in probs.items()) / sum(w[n] for n in probs)
pred = (prob >= THRESHOLD).astype(int)

# ------------------------------------------------------------------
# 4) 제출 파일 저장 & 검증
# ------------------------------------------------------------------
out = pd.DataFrame({"SMILES": test["SMILES"], "output": pred})
out.to_csv(OUTPUT, index=False, encoding="utf-8")
pd.DataFrame({"SMILES": test["SMILES"], "prob_독성無": prob, "output": pred}) \
    .to_csv(OUTPUT.with_name(OUTPUT.stem + "_with_prob.csv"), index=False)

chk = pd.read_csv(OUTPUT)
assert list(chk.columns) == ["SMILES", "output"], "컬럼명/순서 오류"
assert len(chk) == len(test), "행 수 불일치"
assert (chk["SMILES"] == test["SMILES"]).all(), "SMILES 순서 불일치"
assert set(chk["output"].unique()) <= {0, 1}, "output 값 오류"

print("\n" + "=" * 60)
print(f"  제출 파일: {OUTPUT.name}")
print("=" * 60)
print(f"  행 수 {len(chk)} / 컬럼 {list(chk.columns)} / threshold {THRESHOLD:.2f}")
vc = chk["output"].value_counts().sort_index()
for k, v in vc.items():
    print(f"  output {k} ({'독성 無' if k else '독성 有'}): {v}건 ({v/len(chk)*100:.1f}%)")
print(f"  (참고) train label=1 비율 {y.mean()*100:.1f}%")
print("\n[앞 5행]")
print(chk.head().to_string(index=False))
