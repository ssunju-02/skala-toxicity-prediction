# 신약 후보물질 독성 예측 (SKALA 데이터분석 Mini Project)

화합물 SMILES + Fingerprint(ECFP/FCFP/PTFP) + Descriptor 데이터로 독성 여부를 분류하는 모델.
`output` 1 = 독성 無, 0 = 독성 有 / 평가 지표 F1.

## 최종 모델
| 항목 | 내용 |
|---|---|
| Feature | ECFP·FCFP·PTFP 각 1024 bit + Descriptor 4개 + RDKit 2D Descriptor (총 3,293개) |
| 모델 | LightGBM + XGBoost 앙상블 (가중치 0.5 : 0.5, seed 3개 평균) |
| Threshold | 0.41 (OOF 기준 F1 최적값) |
| OOF 성능 | AUC 0.8888 / F1 0.8281 / Accuracy 0.8023 |

설정 전체: [`step4_outputs/final_config.json`](step4_outputs/final_config.json)

## 파이프라인
| 단계 | 스크립트 | 산출물 |
|---|---|---|
| STEP 1. EDA | `step1_eda.py`, `step1_dup_check.py` | `eda_outputs/` |
| STEP 2·3. 베이스라인 | `step2_3_baseline.py` | `baseline_outputs/`, `folds.csv` |
| STEP 4-1. Feature 실험 | `step4_1_feature_experiments.py` | `step4_outputs/best_features.json` |
| STEP 4-2. 튜닝·앙상블 | `step4_2_tuning_ensemble.py` | `step4_outputs/final_config.json` |
| STEP 5. 예측 | `step5_predict.py` | `predict_output_김선주.csv` |

공통 feature 생성 로직은 `features.py` (STEP 4·5 공용).

## 실행
```bash
pip install -r requirements.txt
python step5_predict.py
```
`train.csv` 전체로 최종 모델을 재학습한 뒤 `predict_input.csv`(927건)를 예측해
`predict_output_김선주.csv` (`SMILES`, `output`)를 생성. 확률값 포함 버전은 `predict_output_김선주_with_prob.csv`.

## 예측 결과 요약
- 927건 중 output 1(독성 無) 545건(58.8%), output 0(독성 有) 382건(41.2%)
