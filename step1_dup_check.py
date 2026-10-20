"""
STEP 1 보완. RDKit 정규화(canonical SMILES) 기준 중복 점검
  - 원본 SMILES 문자열은 모두 고유하지만, 같은 분자를 다르게 표기했을 수 있음
  - 3단계로 비교:
      (1) 원본 문자열
      (2) RDKit canonical SMILES (입체 정보 유지)
      (3) RDKit canonical SMILES (입체 정보 제거)  → 거울상·cis/trans 이성질체를 같은 분자로 봄
      (4) (3) + 염/용매 제거(가장 큰 조각만), 전하 중화
  - 각 단계에서 중복 그룹 수, label 충돌 그룹 수 확인
  - 지문(fingerprint) 동일 그룹과의 관계 확인
결과: eda_outputs/dup_check_groups.csv
"""
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")
BASE = Path(__file__).resolve().parent
OUT = BASE / "eda_outputs"

df = pd.read_csv(BASE / "train.csv")
fp_cols = [c for c in df.columns if c.startswith(("ecfp_", "fcfp_", "ptfp_"))]
df = df[["SMILES", "label"] + fp_cols].copy()

frag_chooser = rdMolStandardize.LargestFragmentChooser()
uncharger = rdMolStandardize.Uncharger()


def keys(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None, None, None
    can = Chem.MolToSmiles(mol)
    flat = Chem.MolToSmiles(mol, isomericSmiles=False)
    parent = uncharger.uncharge(frag_chooser.choose(mol))
    std = Chem.MolToSmiles(parent, isomericSmiles=False)
    return can, flat, std


k = df["SMILES"].map(keys)
df["can"] = [x[0] for x in k]
df["flat"] = [x[1] for x in k]
df["std"] = [x[2] for x in k]
df["fp_key"] = pd.util.hash_pandas_object(df[fp_cols], index=False).values
print(f"RDKit 파싱 실패: {df['can'].isna().sum()}건")


def report(col, name):
    g = df.groupby(col)["label"].agg(["size", "nunique"])
    dup = g[g["size"] > 1]
    rows = int(dup["size"].sum())
    print(f"  {name:<36} 고유 {df[col].nunique():>5} | 중복 그룹 {len(dup):>4} | 중복 행 {rows:>4} "
          f"| 제거 시 감소 {rows - len(dup):>4}행 | label 충돌 그룹 {(dup['nunique'] > 1).sum():>3}")
    return dup


print("\n[중복 점검 결과] (전체 8,349행)")
report("SMILES", "(1) 원본 SMILES 문자열")
d_can = report("can", "(2) Canonical SMILES (입체 유지)")
d_flat = report("flat", "(3) Canonical SMILES (입체 제거)")
d_std = report("std", "(4) (3) + 염 제거·전하 중화")
d_fp = report("fp_key", "(참고) Fingerprint 3종 완전 동일")

# 예시 출력
def show(col, dup, n=5):
    for key in list(dup.index[:n]):
        sub = df.loc[df[col] == key, ["SMILES", "label"]]
        print("   ---")
        for _, r in sub.iterrows():
            print(f"   label {r['label']} | {r['SMILES']}")

if len(d_can):
    print("\n[예시] (2) canonical 기준 중복")
    show("can", d_can)
print("\n[예시] (3) 입체 정보 제거 시 중복 (label 충돌 우선)")
show("flat", d_flat.sort_values("nunique", ascending=False))

# 입체 제거 중복 그룹과 FP 동일 그룹의 관계
same_fp = df[df["flat"].isin(d_flat.index)].groupby("flat")["fp_key"].nunique()
print(f"\n입체 제거 중복 그룹 {len(d_flat)}개 중 FP도 완전히 같은 그룹: {(same_fp == 1).sum()}개")

out = df[df["flat"].isin(d_flat.index) | df["std"].isin(d_std.index)][["SMILES", "can", "flat", "std", "label"]]
out.sort_values(["std", "flat"]).to_csv(OUT / "dup_check_groups.csv", index=False)
print(f"저장: {OUT / 'dup_check_groups.csv'} ({len(out)}행)")
