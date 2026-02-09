# 모델 유사도 측정 (Model Similarity Analysis)

## CKA (Centered Kernel Alignment) 분석

두 모델(다른 seed로 학습)이 비슷한 representation을 학습했는지 측정하는 도구입니다.

### 방법론

1. **Feature 추출**: 각 모델의 `stage5` 출력에서 feature map 추출 (64×64×512)
2. **Adaptive Pooling**: 64×64 → 8×8로 축소 (메모리 및 계산 효율성)
3. **Vector 변환**: 각 이미지를 8×8×512 = 32,768 차원 벡터로 표현
4. **CKA 계산**: 두 모델의 feature 유사도를 CKA 지표로 측정

### 데이터 샘플링

- **기본**: `val_split.csv`에서 validation 데이터셋 로드
- **클래스 균형 샘플링**: `--num_samples`를 지정하면 각 클래스에서 동일한 개수의 이미지를 샘플링
  - 예: `--num_samples 1000` → 클래스당 250개씩 균등 샘플링

### CKA 지표 해석

- **Linear CKA**: 선형 커널 사용 (빠르고 안정적)
- **Kernel CKA**: RBF 커널 사용 (비선형 패턴 포착, 느림)

| CKA 범위 | 해석 |
|---------|------|
| ≥ 0.9 | 매우 유사한 representation |
| 0.7 - 0.9 | 유사한 representation |
| 0.5 - 0.7 | 중간 정도 유사 |
| < 0.5 | 다른 representation |

---

## 사용법

### 기본 사용법 (Colab)

```python
# Colab에서 실행 - validation 데이터셋에서 클래스당 250개씩 샘플링
!python -m 모델\ 유사도\ 측정.cka_analysis \
    --ckpt_path_1 "/content/drive/MyDrive/model_seed42/best_model.pt" \
    --ckpt_path_2 "/content/drive/MyDrive/model_seed123/best_model.pt" \
    --shard_root "/content/wds_shards" \
    --save_dir_1 "/content/drive/MyDrive/model_seed42" \
    --output_dir "/content/drive/MyDrive/cka_results" \
    --num_samples 1000 \
    --blocks "2,2,2,3" \
    --dilations "1,1,1,1"
```

### 전체 validation 데이터셋 사용

```python
# num_samples를 0으로 설정하면 전체 validation 데이터셋 사용
!python -m 모델\ 유사도\ 측정.cka_analysis \
    --ckpt_path_1 "/path/to/model1/best_model.pt" \
    --ckpt_path_2 "/path/to/model2/best_model.pt" \
    --save_dir_1 "/path/to/model1" \
    --num_samples 0
```

### 주요 인자

| 인자 | 설명 | 기본값 |
|-----|------|--------|
| `--ckpt_path_1` | 첫 번째 모델 체크포인트 경로 | (필수) |
| `--ckpt_path_2` | 두 번째 모델 체크포인트 경로 | (필수) |
| `--shard_root` | WebDataset shard 경로 | `/content/wds_shards` |
| `--save_dir_1` | 첫 번째 모델 저장 디렉토리 (**val_split.csv** 로딩용) | |
| `--output_dir` | 결과 저장 경로 | `./cka_results` |
| `--num_samples` | 총 샘플 수 (클래스당 num_samples/4개 균등 샘플링, 0=전체 사용) | 1000 |
| `--blocks` | 모델 블록 구성 | `"2,2,2,3"` |
| `--batch_size` | 배치 크기 | 64 |

### 출력 예시

```
============================================================
CKA Analysis Summary
============================================================
Model 1: best_model_seed42.pt
Model 2: best_model_seed123.pt
Number of samples: 1000
Feature dimension: 32768 (8x8x512)
------------------------------------------------------------
Linear CKA:  0.8523
Kernel CKA:  0.8412
------------------------------------------------------------
Per-Class Linear CKA:
  Control     : 0.8634
  SNCA        : 0.8512
  GBA         : 0.8389
  LRRK2       : 0.8557
============================================================

Interpretation:
  → The two models have SIMILAR representations (0.7 ≤ CKA < 0.9)
```

### 결과 파일

`cka_results.json`:
```json
{
  "ckpt_path_1": "/path/to/model1.pt",
  "ckpt_path_2": "/path/to/model2.pt",
  "num_samples": 1000,
  "feature_dim": 32768,
  "linear_cka": 0.8523,
  "kernel_cka": 0.8412,
  "per_class_linear_cka": {
    "Control": 0.8634,
    "SNCA": 0.8512,
    "GBA": 0.8389,
    "LRRK2": 0.8557
  }
}
```

---

## 기술적 세부사항

### CKA 수식

CKA는 두 representation 간의 유사도를 측정하며, 다음과 같이 정의됩니다:

$$\text{CKA}(K, L) = \frac{\text{HSIC}(K, L)}{\sqrt{\text{HSIC}(K, K) \cdot \text{HSIC}(L, L)}}$$

여기서 HSIC (Hilbert-Schmidt Independence Criterion)는:

$$\text{HSIC}(K, L) = \frac{1}{(n-1)^2} \text{tr}(KHLH)$$

- $H = I - \frac{1}{n}\mathbf{1}\mathbf{1}^T$: centering matrix
- $K, L$: 커널 행렬

### 왜 Stage5를 사용하는가?

- SAE 분석에서 `--which_layer stage5_out` 사용 (마지막 resblock 전)
- 고수준 semantic feature를 포함
- Refine block 전이므로 더 일반적인 representation 비교 가능

### 왜 8×8 Pooling?

- 원본 64×64 = 4096 토큰 → 메모리/계산 부담
- 8×8 = 64 토큰으로 축소 → 효율적인 CKA 계산
- 공간 구조 보존하면서 차원 축소

---

## 참고 문헌

- Kornblith, S., Norouzi, M., Lee, H., & Hinton, G. (2019). Similarity of Neural Network Representations Revisited. ICML.
