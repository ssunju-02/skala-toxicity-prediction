"""
STEP 1. 데이터 이해 (EDA) - 신약 후보물질 독성 예측
  - label 분포 (주의: 1 = 독성 無, 0 = 독성 有)
  - 결측치 / 중복 SMILES
  - 4개 descriptor(MolWt, clogp, sa_score, qed)의 label별 분포 비교
  - Fingerprint(ECFP/FCFP/PTFP) 희소성 확인
결과 그림은 eda_outputs/ 폴더에 저장됩니다.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu

plt.rcParams["font.family"] = "AppleGothic"   # macOS 한글 폰트
plt.rcParams["axes.unicode_minus"] = False

BASE = Path(__file__).resolve().parent
OUT = BASE / "eda_outputs"
OUT.mkdir(exist_ok=True)

DESC_COLS = ["MolWt", "clogp", "sa_score", "qed"]
LABEL_NAME = {0: "0 (독성 有)", 1: "1 (독성 無)"}


def section(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


# ------------------------------------------------------------------
# 0. 데이터 로드
# ------------------------------------------------------------------
df = pd.read_csv(BASE / "train.csv")
ecfp_cols = [c for c in df.columns if c.startswith("ecfp_")]
fcfp_cols = [c for c in df.columns if c.startswith("fcfp_")]
ptfp_cols = [c for c in df.columns if c.startswith("ptfp_")]
fp_cols = ecfp_cols + fcfp_cols + ptfp_cols

section("0. 데이터 기본 정보")
print(f"shape: {df.shape}")
print(f"ECFP {len(ecfp_cols)}개 / FCFP {len(fcfp_cols)}개 / PTFP {len(ptfp_cols)}개 / descriptor {len(DESC_COLS)}개")
print("\n[dtype 분포]")
print(df.dtypes.value_counts())
print("\n[앞 5행 - 주요 컬럼]")
print(df[["SMILES"] + DESC_COLS + ["label"]].head())

# ------------------------------------------------------------------
# 1. label 분포
# ------------------------------------------------------------------
section("1. label 분포 (1 = 독성 無, 0 = 독성 有)")
vc = df["label"].value_counts().sort_index()
ratio = df["label"].value_counts(normalize=True).sort_index()
for k in vc.index:
    print(f"  label {LABEL_NAME[k]:<12}: {vc[k]:>5}건 ({ratio[k]*100:.2f}%)")
imb = vc.max() / vc.min()
print(f"  불균형 비율(다수/소수): {imb:.2f} → "
      f"{'불균형 있음 → Stratified K-Fold, class_weight, AUC/F1 지표 고려' if imb > 1.5 else '비교적 균형'}")

fig, ax = plt.subplots(figsize=(5, 4))
sns.countplot(x=df["label"].map(LABEL_NAME), ax=ax, palette=["#d9534f", "#5cb85c"],
              order=[LABEL_NAME[0], LABEL_NAME[1]])
for p in ax.patches:
    ax.annotate(f"{int(p.get_height())}\n({p.get_height()/len(df)*100:.1f}%)",
                (p.get_x() + p.get_width() / 2, p.get_height()), ha="center", va="bottom")
ax.set_title("label 분포")
ax.set_xlabel("")
ax.set_ylim(0, vc.max() * 1.2)
fig.tight_layout()
fig.savefig(OUT / "01_label_distribution.png", dpi=120)
plt.close(fig)

# ------------------------------------------------------------------
# 2. 결측치 & 중복
# ------------------------------------------------------------------
section("2. 결측치 / 중복 확인")
na = df.isna().sum()
print(f"  결측치가 있는 컬럼 수: {(na > 0).sum()}  (전체 결측 셀 수: {na.sum()})")
if na.sum() > 0:
    print(na[na > 0].sort_values(ascending=False).head(20))

dup_rows = df.duplicated().sum()
dup_smiles_mask = df["SMILES"].duplicated(keep=False)
n_dup_smiles = df["SMILES"].duplicated().sum()
print(f"  완전 중복 행 수: {dup_rows}")
print(f"  고유 SMILES 수: {df['SMILES'].nunique()} / 전체 {len(df)}")
print(f"  중복 SMILES (첫 등장 제외) 수: {n_dup_smiles}")

if n_dup_smiles > 0:
    g = df[dup_smiles_mask].groupby("SMILES")["label"].agg(["count", "nunique"])
    conflict = g[g["nunique"] > 1]
    print(f"  중복 SMILES 그룹 수: {len(g)},  그 중 label이 서로 다른 그룹: {len(conflict)}")
    if len(conflict):
        print(conflict.head(10))

# fingerprint가 완전히 같은데 SMILES가 다른 경우 (모델 입장에서는 구분 불가)
fp_dup = df.duplicated(subset=fp_cols, keep=False)
print(f"  Fingerprint 3종이 완전히 동일한 행 수: {fp_dup.sum()}")
if fp_dup.sum() > 0:
    key = pd.util.hash_pandas_object(df[fp_cols], index=False)
    gg = df[fp_dup].assign(_k=key[fp_dup]).groupby("_k")["label"].nunique()
    print(f"    └ 동일 FP 그룹 {len(gg)}개 중 label 충돌 그룹: {(gg > 1).sum()}")

# 값 범위 확인 (FP가 진짜 0/1인지)
fp_vals = np.unique(df[fp_cols].values)
print(f"  Fingerprint 값 종류: {fp_vals[:10]}")

# ------------------------------------------------------------------
# 3. Descriptor 분포 (label별)
# ------------------------------------------------------------------
section("3. Descriptor 기술통계 (label별)")
print(df[DESC_COLS].describe().T.round(3))
print()
grp = df.groupby("label")[DESC_COLS].agg(["mean", "median", "std"]).T.round(3)
grp.columns = [LABEL_NAME[c] for c in grp.columns]
print(grp)

print("\n[Mann-Whitney U 검정: label 0 vs 1 분포 차이]")
rows = []
for c in DESC_COLS:
    a = df.loc[df.label == 0, c].dropna()
    b = df.loc[df.label == 1, c].dropna()
    stat, p = mannwhitneyu(a, b)
    auc = stat / (len(a) * len(b))  # P(독성有 값 > 독성無 값)
    rows.append({"feature": c, "median_독성有": a.median(), "median_독성無": b.median(),
                 "p_value": p, "AUC(단일변수)": max(auc, 1 - auc)})
stat_df = pd.DataFrame(rows).round(4)
print(stat_df.to_string(index=False))
print("  * AUC(단일변수): 해당 descriptor 하나만으로 두 클래스를 얼마나 구분하는지 (0.5=무의미)")

# 히스토그램 + KDE
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for ax, c in zip(axes.ravel(), DESC_COLS):
    sns.histplot(data=df.assign(label=df["label"].map(LABEL_NAME)), x=c, hue="label",
                 kde=True, stat="density", common_norm=False, ax=ax,
                 palette={LABEL_NAME[0]: "#d9534f", LABEL_NAME[1]: "#5cb85c"}, alpha=0.4)
    ax.set_title(f"{c} 분포 (label별)")
fig.tight_layout()
fig.savefig(OUT / "02_descriptor_hist_by_label.png", dpi=120)
plt.close(fig)

# 박스플롯
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, c in zip(axes, DESC_COLS):
    sns.boxplot(x=df["label"].map(LABEL_NAME), y=df[c], ax=ax,
                order=[LABEL_NAME[0], LABEL_NAME[1]], palette=["#d9534f", "#5cb85c"])
    ax.set_title(c)
    ax.set_xlabel("")
fig.tight_layout()
fig.savefig(OUT / "03_descriptor_box_by_label.png", dpi=120)
plt.close(fig)

# 구간별 독성 비율 (예: clogp가 높을수록 독성이 많은가?)
print("\n[구간(5분위)별 독성 有 비율]")
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, c in zip(axes, DESC_COLS):
    bins = pd.qcut(df[c], q=5, duplicates="drop")
    tox_rate = (1 - df["label"]).groupby(bins, observed=True).mean()
    print(f"  {c}:")
    for k, v in tox_rate.items():
        print(f"     {str(k):<22} 독성 有 비율 {v*100:5.1f}%")
    ax.bar(range(len(tox_rate)), tox_rate.values * 100, color="#d9534f")
    ax.set_xticks(range(len(tox_rate)))
    ax.set_xticklabels([f"Q{i+1}" for i in range(len(tox_rate))])
    ax.axhline((1 - df["label"]).mean() * 100, ls="--", c="gray", label="전체 평균")
    ax.set_title(f"{c} 5분위별 독성 有 비율(%)")
    ax.legend()
fig.tight_layout()
fig.savefig(OUT / "04_descriptor_quantile_toxic_rate.png", dpi=120)
plt.close(fig)

# 상관관계
corr = df[DESC_COLS + ["label"]].corr()
print("\n[Descriptor + label 상관계수]")
print(corr.round(3))
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Descriptor 상관관계")
fig.tight_layout()
fig.savefig(OUT / "05_descriptor_corr.png", dpi=120)
plt.close(fig)

# ------------------------------------------------------------------
# 4. Fingerprint 희소성 & label 관련 비트
# ------------------------------------------------------------------
section("4. Fingerprint 희소성")
summary = []
for name, cols in [("ECFP", ecfp_cols), ("FCFP", fcfp_cols), ("PTFP", ptfp_cols)]:
    X = df[cols]
    bit_freq = X.mean()
    summary.append({
        "FP": name,
        "1의 비율(전체 셀)": X.values.mean(),
        "분자당 평균 on-bit 수": X.sum(axis=1).mean(),
        "항상 0인 비트": int((bit_freq == 0).sum()),
        "항상 1인 비트": int((bit_freq == 1).sum()),
        "1% 미만 등장 비트": int((bit_freq < 0.01).sum()),
    })
fp_sum = pd.DataFrame(summary).round(4)
print(fp_sum.to_string(index=False))

# label별 on-bit 비율 차이가 큰 비트 Top 15
print("\n[label 간 등장 비율 차이가 큰 비트 Top 15]")
p0 = df.loc[df.label == 0, fp_cols].mean()
p1 = df.loc[df.label == 1, fp_cols].mean()
diff = (p0 - p1).rename("독성有-독성無")
top = pd.concat([p0.rename("독성有 비율"), p1.rename("독성無 비율"), diff], axis=1)
top = top.reindex(diff.abs().sort_values(ascending=False).index).head(15).round(3)
print(top)

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for a, (name, cols) in zip(ax, [("ECFP", ecfp_cols), ("FCFP", fcfp_cols), ("PTFP", ptfp_cols)]):
    on_bits = df[cols].sum(axis=1)
    sns.histplot(x=on_bits, hue=df["label"].map(LABEL_NAME), ax=a, stat="density",
                 common_norm=False, element="step",
                 palette={LABEL_NAME[0]: "#d9534f", LABEL_NAME[1]: "#5cb85c"})
    a.set_title(f"{name}: 분자당 on-bit 수")
fig.tight_layout()
fig.savefig(OUT / "06_fp_onbits_by_label.png", dpi=120)
plt.close(fig)

# SMILES 길이
df["smiles_len"] = df["SMILES"].str.len()
print("\n[SMILES 길이 (label별 평균)]")
print(df.groupby("label")["smiles_len"].describe().round(1))

section("완료")
print(f"그래프 저장 위치: {OUT}")
for f in sorted(OUT.glob("*.png")):
    print("  -", f.name)
