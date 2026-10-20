"""
공통 Feature 모듈 - STEP 4(성능 개선), STEP 5(예측)에서 동일하게 사용
  - 기본 feature: ECFP / FCFP / PTFP (각 1024 bit) + descriptor 4개
  - 추가 feature (RDKit): 2D descriptor 217개, MACCS keys 167 bit
"""
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, MACCSkeys

RDLogger.DisableLog("rdApp.*")

BASE = Path(__file__).resolve().parent
CACHE = BASE / "feature_cache"
CACHE.mkdir(exist_ok=True)

DESC_COLS = ["MolWt", "clogp", "sa_score", "qed"]


def fp_groups(df):
    """원본 데이터의 컬럼 그룹"""
    return {
        "ecfp": [c for c in df.columns if c.startswith("ecfp_")],
        "fcfp": [c for c in df.columns if c.startswith("fcfp_")],
        "ptfp": [c for c in df.columns if c.startswith("ptfp_")],
        "desc": DESC_COLS,
    }


def _calc_one(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None, None
    desc = {}
    for name, fn in Descriptors._descList:
        try:
            desc[f"rd_{name}"] = fn(mol)
        except Exception:
            desc[f"rd_{name}"] = np.nan
    maccs = np.array(list(MACCSkeys.GenMACCSKeys(mol)), dtype=np.int8)
    return desc, maccs


def rdkit_features(smiles, cache_name):
    """SMILES → RDKit descriptor(rd_*) + MACCS(maccs_*) DataFrame. 결과는 캐시 파일로 저장."""
    path = CACHE / f"{cache_name}_rdkit.pkl"
    if path.exists():
        cached = pd.read_pickle(path)
        if cached.attrs.get("smiles") == list(smiles):   # 같은 SMILES 목록일 때만 캐시 사용
            return cached

    desc_rows, maccs_rows, n_fail = [], [], 0
    for smi in smiles:
        d, m = _calc_one(smi)
        if d is None:
            n_fail += 1
            d, m = {}, np.full(167, np.nan)
        desc_rows.append(d)
        maccs_rows.append(m)
    desc = pd.DataFrame(desc_rows)
    maccs = pd.DataFrame(np.vstack(maccs_rows), columns=[f"maccs_{i}" for i in range(167)])
    out = pd.concat([desc, maccs], axis=1)
    # 무한대/비정상적으로 큰 값은 NaN 처리 (트리 모델은 NaN을 그대로 처리)
    out = out.replace([np.inf, -np.inf], np.nan)
    num = out.select_dtypes(include="number").columns
    out[num] = out[num].mask(out[num].abs() > 1e10)
    out = out.astype(np.float32)
    out.attrs["smiles"] = list(smiles)
    out.to_pickle(path)
    print(f"  RDKit feature 계산 완료: {out.shape}, SMILES 파싱 실패 {n_fail}건 → {path.name}")
    return out


def build_features(df, cache_name):
    """원본 df + RDKit feature를 합친 전체 feature DataFrame과 그룹 dict 반환"""
    groups = fp_groups(df)
    rd = rdkit_features(df["SMILES"].tolist(), cache_name)
    groups["rdkit"] = [c for c in rd.columns if c.startswith("rd_")]
    groups["maccs"] = [c for c in rd.columns if c.startswith("maccs_")]
    base_cols = groups["ecfp"] + groups["fcfp"] + groups["ptfp"] + groups["desc"]
    feats = pd.concat([df[base_cols].reset_index(drop=True).astype(np.float32),
                       rd.reset_index(drop=True)], axis=1)
    return feats, groups


def select_cols(groups, names):
    cols = []
    for n in names:
        cols += groups[n]
    return cols


BIT_PREFIX = ("ecfp_", "fcfp_", "ptfp_", "maccs_")


def drop_rare_bits(X_tr, cols, lo=0.01, hi=0.99):
    """등장 빈도 1% 미만 / 99% 초과인 0-1 bit 제거 (연속형 변수는 상수만 제거)"""
    keep = []
    for c in cols:
        if c.startswith(BIT_PREFIX):
            if lo <= X_tr[c].mean() <= hi:
                keep.append(c)
        elif X_tr[c].nunique(dropna=True) > 1:
            keep.append(c)
    return keep


def top_k_importance(X_tr, y_tr, cols, k, seed=42):
    """LightGBM gain importance 상위 k개 컬럼 (train 데이터로만 계산)"""
    from lightgbm import LGBMClassifier
    m = LGBMClassifier(n_estimators=300, learning_rate=0.05, colsample_bytree=0.3,
                       importance_type="gain", random_state=seed, n_jobs=-1, verbose=-1)
    m.fit(X_tr[cols], y_tr)
    imp = pd.Series(m.feature_importances_, index=cols)
    return imp.sort_values(ascending=False).index[:k].tolist()


def apply_selection(X_tr, y_tr, cols, selection):
    """best_features.json의 selection 값(None / 'rare' / 정수 k)에 맞춰 컬럼 선택"""
    if selection is None:
        return list(cols)
    if selection == "rare":
        return drop_rare_bits(X_tr, cols)
    return top_k_importance(X_tr, y_tr, cols, int(selection))
