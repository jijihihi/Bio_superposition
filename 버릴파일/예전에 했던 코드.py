############################################# PCA 먼저 하고 그 이후에 내가 하던거 다 하자. PCA를 먼저하는게 scRNA 표준이니까 따라야지. ##########################

#################### 단순 GAP을 이용해서 이미지에게 벡터 할당해줘서 affinity kNN을 만든다면 NMI와 타우 상관계수가 있을까? GAP에다 amount 나눠줌. 여기서 amount는 피처맵에서 attention>0인 그리드 피처맵 모두에 대해 union. 결과 좀 잘나옴

########### 유니온 보정이 효과가 없어서 GAP L1 L2 보정으로 그런 효과.

##### 3차원에 SE 어떻게 됐는지 보기. k=90

##### 이미지 뽑을때 가에 있는 이미지 x_900 혹은 y_900 포함되면 제외.

##### 피처맵 선택 기준으로 accuracy로 잡음 Top_k = 200

### DPT에서 t=0.5

## 예전 코랩 코드.

import os
import random
import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import torch
import torch.nn as nn
from PIL import Image
from scipy import stats
from scipy.sparse import csgraph
from scipy.sparse.linalg import \
    eigsh  # 대칭 행렬(Symmetric Matrix)에 한해서는 eigsh가 일반 eigs보다 정확하고 안정적. 큰거 몇개만 알면 될때 eigsh 유리.
from scipy.spatial.distance import pdist, squareform
from scipy.stats import kendalltau
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors
from torch.nn.utils import weight_norm
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

warnings.filterwarnings("ignore")

# ==========================================================================================
# 0. 사용자 설정
# ==========================================================================================
BASE_DIR = "/content/drive/MyDrive/New_data/supcon_2_instance_norm_v3_darkfilter_Weight_Normalization_stride"
OUTPUT_DIR = os.path.join(BASE_DIR, "feature_analysis")

MODEL_PATH = os.path.join(BASE_DIR, "supcon_best_model.pt")
ACCURACY_CSV_PATH = "/content/single_feature_threshold_accuracies_g_eq_1_stride.csv"  ### 지금은 SAE로 뽑아낸 dead neuron이 아닌 개념 하면 될듯.
INDEX_MAPPING_FILE_PATH = "/content/matrix_index_total.csv"
APOPTOSIS_FILE_PATH = "/content/apoptosis_indices_with_NMI_11.27.csv"

SEED = 42
SAMPLES_PER_CLASS = 4000
ACC_THRESHOLD = 0.75
IMAGE_SIZE = 180
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
t_scale = 0.5
n_neighbors = 90

min_dim = 2
max_dim = 11

TOP_K = 200  #### SNCA와 control에서 각각 사용할 피처맵 개수.


# 재현성 설정
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ==========================================================================================
# 1. 모델 및 데이터셋 정의 (기존과 동일)
# ==========================================================================================
class CNN4(nn.Module):
    def __init__(self):
        super(CNN4, self).__init__()
        # 1. Stride를 2로 주어 Downsampling 수행
        self.conv1 = weight_norm(nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1))
        self.relu1 = nn.ReLU()

        self.conv2 = weight_norm(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1))
        self.relu2 = nn.ReLU()

        self.conv3 = weight_norm(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        )
        self.relu3 = nn.ReLU()

        self.conv4 = weight_norm(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        )
        self.relu4 = nn.ReLU()

        # [수정 1] 시각화/분석용이므로 forward에서 GAP를 통과시키지 않기 위해 주석 처리하거나 사용하지 않음
        # self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.relu1(self.conv1(x))  # 180 -> 90
        x = self.relu2(self.conv2(x))  # 90 -> 45
        x = self.relu3(self.conv3(x))  # 45 -> 45
        x = self.relu4(self.conv4(x))  # 45 -> 45 (피처맵)

        # [수정 2] GAP와 Flatten을 제거하고 4차원 피처맵 그대로 반환 (B, 512, 45, 45)
        return x


class InstanceNormalize(object):
    def __call__(self, tensor):
        mean = torch.mean(tensor, dim=[1, 2], keepdim=True)
        std = torch.std(tensor, dim=[1, 2], keepdim=True)
        return (tensor - mean) / (std + 1e-8)


class SimpleImageDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filepath = row["path"]
        label = row["label"] if "label" in row else 0
        if label == "control":
            label_idx = 0
        elif label == "SNCA":
            label_idx = 1
        else:
            label_idx = 0

        try:
            image = Image.open(filepath).convert("RGB")
            return (
                self.transform(image),
                torch.tensor(label_idx, dtype=torch.long),
                filepath,
            )
        except Exception as e:
            return torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE), torch.tensor(-1), filepath


# ==========================================================================================
# 2. 데이터 준비 및 경로 재구성 함수
# ==========================================================================================
def reconstruct_path_for_nmi(row):
    try:
        condition = row["Condition"]
        exp_id_num = row["ExperimentID"]
        formatted_exp_id = ""

        if condition == "control":
            formatted_exp_id = str(exp_id_num).zfill(6)
        elif condition == "SNCA":
            base_id = str(exp_id_num).zfill(6)
            # 파일 경로 규칙에 맞게 수정 필요 (사용자 환경에 맞게)
            formatted_exp_id = f"{base_id}"

        filename_key = row["FilenameKey"].lstrip("/")
        match = re.search(r"(_[xy]\d+.*)", filename_key)
        if match:
            base_name = filename_key[: match.start()]
            coords = match.group(1)
            filename_final = f"{base_name}_Composite_RGB{coords}.tif"
        else:
            base_name = filename_key
            filename_final = f"{base_name}_Composite_RGB.tif"
        return f"/content/{condition}/{formatted_exp_id}/{filename_final}"
    except:
        return None


# 데이터 로드
df_index_total = pd.read_csv(INDEX_MAPPING_FILE_PATH)

print(f"Total images loaded: {len(df_index_total)}")

# ---------------------------------------------------------------------------
# [FILTERING] x_900 또는 y_900이 포함된 이미지(가장자리) 제거
# ---------------------------------------------------------------------------
# 정규표현식 'x_900|y_900'을 사용하여 해당 문자열이 포함된 행을 찾고(mask),
# '~' (NOT) 연산자로 포함되지 않은 행만 남깁니다.
edge_mask = df_index_total["path"].astype(str).str.contains("x_900|y_900", regex=True)
n_removed = edge_mask.sum()
print(n_removed)

df_index_total = df_index_total[~edge_mask].reset_index(drop=True)

print(f"Removed {n_removed} edge images (x_900/y_900).")
print(f"Remaining valid images: {len(df_index_total)}")
# ---------------------------------------------------------------------------

# 그 다음 샘플링 진행 (이제 깨끗한 데이터에서 샘플링함)
df_sampled = (
    df_index_total.groupby("label", group_keys=False)
    .apply(lambda x: x.sample(min(len(x), SAMPLES_PER_CLASS), random_state=SEED))
    .reset_index(drop=True)
)

print(f"Sampled Data (Cleaned): {len(df_sampled)}")

# ==========================================================================================
# 3. 모델 로드 및 Active-Area Normalized Feature 추출
# ==========================================================================================
model = CNN4().to(DEVICE)
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
if "encoder." in list(state_dict.keys())[0]:
    state_dict = {
        k.replace("encoder.", ""): v
        for k, v in state_dict.items()
        if k.startswith("encoder.")
    }
model.load_state_dict(state_dict, strict=False)
model.eval()

print("Forcing weight_g to 1.0 and removing Weight Norm...")

for m in model.modules():
    if isinstance(m, nn.Conv2d):
        # [핵심 추가] g 파라미터가 있다면, 먼저 1로 강제 초기화합니다.
        if hasattr(m, "weight_g"):
            m.weight_g.data.fill_(1.0)

        # 그 다음 WN을 제거하면, 1.0 * (v / ||v||) 상태로 합쳐집니다.
        try:
            nn.utils.remove_weight_norm(m)
        except:
            pass

transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        InstanceNormalize(),
    ]
)
dataset = SimpleImageDataset(df_sampled, transform)
loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)

print("Extracting Features with Active-Area Normalization (Sum / Count>0)...")
gap_feats = []
valid_indices = []  # [중요] 성공적으로 로드된 인덱스를 저장할 리스트

# enumerate를 사용하여 원본 데이터프레임의 인덱스(idx)나 루프 카운트(i)를 추적해야 함
# 하지만 DataLoader는 배치를 섞지 않는다면(shuffle=False) 순서대로 나오므로
# 전체 카운터를 써서 추적합니다.

current_idx = 0


with torch.no_grad():
    for imgs, labels, _ in loader:
        batch_size = imgs.size(0)
        imgs = imgs.to(DEVICE)

        # 모델 통과
        out = model(imgs)  # (B, 512, 45, 45)

        # [수정] Amount 정규화 제거 -> Standard GAP (단순 평균) 적용
        # active_count(활성 개수)로 나누지 않고, 그냥 전체 평균을 구합니다.
        # 이렇게 하면 '활성 영역의 크기(양)'가 값에 그대로 반영됩니다.
        normalized_feat = torch.mean(out, dim=[2, 3])

        # 만약 평균(Mean) 대신 합계(Sum)를 쓰고 싶다면 아래 줄을 쓰세요 (결과 경향성은 동일함)
        # normalized_feat = torch.sum(out, dim=[2, 3])

        # CPU로 이동
        batch_feats = normalized_feat.cpu().numpy()

        # [핵심 수정] 배치 내부에서 유효한(0이 아닌) 이미지만 골라내기
        # SimpleImageDataset이 실패시 0을 반환하므로, max()가 0인 것을 찾습니다.

        # 배치 내 각 이미지별로 확인
        for i in range(batch_size):
            # 원본 이미지가 0인지 확인 (로드 실패 여부)
            if imgs[i].max().item() == 0:
                pass  # 건너뜀
            else:
                gap_feats.append(batch_feats[i])
                valid_indices.append(current_idx + i)  # 유효한 원본 인덱스 저장

        current_idx += batch_size

# 전체 데이터 병합
if len(gap_feats) > 0:
    X_gap = np.array(gap_feats)  # (N_valid, 512)
    print(f"Features Extracted Shape: {X_gap.shape}")

    # [핵심] 데이터프레임도 유효한 행만 남기고 동기화 (Sync)
    print(f"Original DataFrame: {len(df_sampled)}")
    df_sampled = df_sampled.iloc[valid_indices].reset_index(drop=True)
    print(f"Filtered DataFrame: {len(df_sampled)}")

    if len(df_sampled) != len(X_gap):
        raise ValueError(
            f"치명적 오류: 피처 개수({len(X_gap)})와 데이터프레임 개수({len(df_sampled)}) 불일치!"
        )

else:
    raise ValueError("추출된 피처가 없습니다. 데이터셋 경로나 이미지를 확인하세요.")

# Feature Filtering (Acc >= 75%)
print("\n>>> Performing Balanced Feature Selection...")

if os.path.exists(ACCURACY_CSV_PATH):
    df_acc = pd.read_csv(ACCURACY_CSV_PATH)

    # 1. 컬럼 이름 확인 (첫 번째 열이 인덱스라고 가정)
    # 만약 인덱스 컬럼명이 'Unnamed: 0'이나 'idx'라면 그걸 쓰세요.
    # 여기서는 첫 번째 열(iloc[:, 0])을 인덱스로 간주하고 정렬합니다.
    first_col_name = df_acc.columns[0]

    # 2. 인덱스 기준 오름차순 정렬 (0, 1, 2, ... 511)
    df_acc = df_acc.sort_values(by=first_col_name).reset_index(drop=True)

    # 3. 정렬된 상태에서 정확도 값 추출
    # 두 번째 열이 정확도라면:
    accs = df_acc.iloc[:, 1].values

    print("✅ Accuracy DataFrame sorted by Index. Shape:", accs.shape)
    print("   First 5 accs:", accs[:5])  # 0,1,2,3,4번 피처의 정확도인지 확인
else:
    print("accuracy 파일이 없습니다.")

# 1. 정확도 로드


snca_indices = np.where((df_sampled["label"] == "SNCA") | (df_sampled["label"] == 1))[0]
ctrl_indices = np.where(
    (df_sampled["label"] == "control") | (df_sampled["label"] == 0)
)[0]

# 2. 각 그룹별 평균 계산 (Gap Feature 기준)
# X_gap Shape: (Total_Samples, 512)
snca_mean = np.mean(X_gap[snca_indices], axis=0)  # (512,)
ctrl_mean = np.mean(X_gap[ctrl_indices], axis=0)  # (512,)

# 2. 방향성(Sign) 계산: 단순히 평균이 어디가 더 큰지만 봅니다 (+1 or -1)
direction = np.sign(snca_mean - ctrl_mean)

# 3. "Signed Accuracy" 생성
# SNCA에서 높으면 +0.8, Control에서 높으면 -0.8
signed_acc = accs * direction

# 4. Top k 개 선정.


# Control 대표: 값이 가장 작은 음수 (예: -0.9, -0.8 ...) -> 앞에서부터 K개
ctrl_top_idx = np.argsort(signed_acc)[:TOP_K]
print(ctrl_top_idx)

# SNCA 대표: 값이 가장 큰 양수 (예: +0.9, +0.8 ...) -> 뒤에서부터 K개
snca_top_idx = np.argsort(signed_acc)[-TOP_K:]
print(snca_top_idx)

# 5. 인덱스 합치기 (기존 동일)
balanced_indices = np.concatenate([snca_top_idx, ctrl_top_idx])
balanced_indices = np.unique(balanced_indices)

# 6. 최종 선택
X_selected = X_gap[:, balanced_indices]
print(f"Selected Top {TOP_K} form SNCA & Top {TOP_K} from Control.")
print(f"Final Feature Shape: {X_selected.shape}")
# (선택 사항) 어떤 피처가 뽑혔는지 확인
print(f"SNCA Top 5 Indices: {snca_top_idx[-TOP_K:]}")
print(f"Control Top 5 Indices: {ctrl_top_idx[:TOP_K]}")

# ==========================================================================================
# 4. 차원별 검증 및 Weyl's Law 분석 수행
# ==========================================================================================

# 테스트할 설정 (이름, 거리척도, 피처정규화방식)
TEST_CONFIGS = [
    ("L1", "cityblock", "none"),
    ("L1_log", "cityblock", "log"),
    ("L1_median", "cityblock", "median"),
    ("L1_log_median", "cityblock", "median_log"),  # (추천) 로그 후 미디언
    ("L1_IQR", "cityblock", "IQR"),
    ("L1_log_IQR", "cityblock", "log_IQR"),  # (추천) 로그 후 IQR
    ("L2", "euclidean", "none"),
    ("L2_log", "euclidean", "log"),
    ("L2_std", "euclidean", "std"),
    ("L2_log_std", "euclidean", "std_log"),
    ("L2_IQR", "euclidean", "IQR"),
    ("Correlation", "correlation", "none"),
    ("L2Norm_Cosine", "cosine", "sample_L2"),
    ("L2Norm_euclidean", "euclidean", "sample_L2"),
    ("L1_norm", "cityblock", "sample_L1"),
    ("L1_norm_cosine", "cosine", "sample_L1"),
    ("Cosine", "cosine", "none"),
    ("Log_Cosine", "cosine", "log"),
    ("Log_IQR_L2Vector", "euclidean", "log_IQR_sample_L2"),
    ("Log_Median_L2Vector", "euclidean", "log_median_sample_L2"),
    ("Log_IQR_L2Vector", "cosine", "log_IQR_sample_L2"),
    ("Log_Median_L2Vector", "cosine", "log_median_sample_L2"),
    ("Log_IQR_L1Vector", "cityblock", "log_IQR_sample_L1"),
    ("Log_Median_L1Vector", "cityblock", "log_std_sample_L1"),
    ("Median_L1Vector", "cityblock", "median_sample_L1"),
    ("std_L2Vector_euclidean", "euclidean", "std_sample_L2"),
    ("std_L2Vector_cosine", "cosine", "std_sample_L2"),
    ("log_median_sample_L1", "cityblock", "log_median_sample_L1"),
    ("log_std_sample_L2", "cosine", "log_std_sample_L2"),
    ("Hellinger_transformation", "euclidean", "hellinger"),
    ("Hellinger_transformatio_after_log", "euclidean", "log_hellinger"),
    ("Hellinger_transformatio_after_log_median", "euclidean", "log_median_hellinger"),
    ("Hellinger_transformation_median", "euclidean", "median_hellinger"),
]


# ----------------------------------------------------------------------
# [A] NMI 데이터 준비 (기존 코드 유지)
# ----------------------------------------------------------------------
df_nmi = pd.read_csv(APOPTOSIS_FILE_PATH)
df_nmi["reconstructed_path"] = df_nmi.apply(reconstruct_path_for_nmi, axis=1)
df_final = pd.merge(
    df_sampled,
    df_nmi[["reconstructed_path", "NMI_Score"]],
    left_on="path",
    right_on="reconstructed_path",
    how="left",
)

is_snca = (df_final["label"] == "SNCA") | (df_final["label"] == 1)


has_nmi = df_final["NMI_Score"].notna()
valid_snca_mask = is_snca & has_nmi
nmi_snca_clean = df_final.loc[valid_snca_mask, "NMI_Score"].values

print(f"Valid SNCA Samples for Validation: {len(nmi_snca_clean)}")
# ----------------------------------------------------------------------
# [B] Weyl's Law (Log-Log Fit) & Tau Analysis Loop
# ----------------------------------------------------------------------
results_summary = []

# PCA 차원 수 설정 (리뷰어 방어용)
N_COMPONENTS_PCA = 50

for name, metric, norm_method in TEST_CONFIGS:
    print(f"\n" + "=" * 80)
    print(f"Processing Configuration: {name} (Metric: {metric}, Norm: {norm_method})")
    print("=" * 80)

    # 1. 피처 정규화 (기존 로직 수행)
    X_norm = X_selected.copy()

    if "log" in norm_method:
        X_norm = np.log1p(X_norm)

    if "IQR" in norm_method:
        q1 = np.percentile(X_norm, 25, axis=0)
        q3 = np.percentile(X_norm, 75, axis=0)
        iqr = q3 - q1
        iqr[iqr == 0] = 1.0
        X_norm = X_norm / iqr

    if "median" in norm_method:
        medians = np.median(X_norm, axis=0)
        medians[medians == 0] = 1.0
        X_norm = X_norm / medians

    if "std" in norm_method:
        stds = np.std(X_norm, axis=0)
        stds[stds == 0] = 1.0
        X_norm = X_norm / stds

    if "sample_L2" in norm_method:
        norms = np.linalg.norm(X_norm, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        X_norm = X_norm / norms
        print("   -> Applied Sample-wise L2 Normalization.")

    if "sample_L1" in norm_method:
        norms = np.linalg.norm(X_norm, ord=1, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        X_norm = X_norm / norms
        print("   -> Applied Sample-wise L1 Normalization.")

    if "hellinger" in norm_method:
        X_norm = X_norm / (np.sum(X_norm, axis=1, keepdims=True) + 1e-8)
        X_norm = np.sqrt(X_norm)
        print("   -> Applied Hellinger Transform.")

    # ======================================================================
    # [추가됨] PCA Dimension Reduction (50차원)
    # ======================================================================
    # 다중공선성을 제거하고 주요 변동성만 남겨 공정한 비교 수행
    current_dim = X_norm.shape[1]

    if current_dim > N_COMPONENTS_PCA:
        print(
            f"   -> Applying PCA: Reducing dimensions from {current_dim} to {N_COMPONENTS_PCA}..."
        )

        pca = PCA(n_components=N_COMPONENTS_PCA, random_state=SEED)
        X_norm = pca.fit_transform(X_norm)

        explained_var = np.sum(pca.explained_variance_ratio_)
        print(f"      [PCA Info] Retained Variance: {explained_var:.4f}")
    else:
        print(
            f"   -> Skipping PCA: Feature count ({current_dim}) <= {N_COMPONENTS_PCA}"
        )

    # ----------------------------------------------------------------------
    # [이후 과정은 PCA된 X_norm을 사용하므로 동일]
    # 2. kNN Graph & Laplacian
    # ----------------------------------------------------------------------
    try:
        # PCA 후에는 데이터가 회전되므로 보통 Euclidean이나 Cosine을 사용합니다.
        # Metric이 'correlation'인 경우 PCA 후에는 의미가 모호해질 수 있으나,
        # 사용자가 지정한 metric을 그대로 따릅니다.
        knn_metric = "cosine" if metric == "correlation" else metric

        nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric=knn_metric).fit(X_norm)
        distances, indices = nbrs.kneighbors(X_norm)

        # Adaptive Gaussian Kernel
        N = X_norm.shape[0]
        sigmas = distances[:, -1]
        W = np.zeros((N, N))

        for i in range(N):
            for j_idx, dist in zip(indices[i], distances[i]):
                if i == j_idx:
                    continue
                scale = sigmas[i] * sigmas[j_idx]
                weight = np.exp(-(dist**2) / (scale if scale > 0 else 1.0))
                W[i, j_idx] = weight
                W[j_idx, i] = weight

        # Normalized Laplacian
        D = np.diag(np.sum(W, axis=1))
        with np.errstate(divide="ignore"):
            D_inv_sqrt = np.power(np.diag(D), -0.5)
        D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0.0
        D_inv_sqrt = np.diag(D_inv_sqrt)
        L_mat = np.eye(N) - D_inv_sqrt @ W @ D_inv_sqrt

    except Exception as e:
        print(f"   Error in Graph Construction: {e}")
        continue

    # [cite_start]3. Eigen Decomposition [cite: 5]
    try:
        if hasattr(L_mat, "toarray"):
            L_mat = L_mat.toarray()
        evals, evecs = np.linalg.eigh(L_mat)
        idx = np.argsort(evals)
        evals = evals[idx]
        evecs = evecs[:, idx]

        # Sign Flip
        control_indices = df_sampled[df_sampled["label"].isin(["control", 0])].index
        if len(control_indices) > 0 and np.mean(evecs[control_indices, 1]) > 0:
            evecs[:, 1] = -evecs[:, 1]

    except Exception as e:
        print(f"   Error in Eigen Decomposition: {e}")
        continue

    # 4. Weyl's Law Fitting
    fit_start, fit_end = 2, 20
    k_indices = np.arange(fit_start, fit_end + 1)
    lambda_vals = evals[fit_start : fit_end + 1]
    valid_mask = lambda_vals > 1e-9

    if np.sum(valid_mask) > 3:
        log_k = np.log(k_indices[valid_mask]).reshape(-1, 1)
        log_lambda = np.log(lambda_vals[valid_mask]).reshape(-1, 1)
        reg = LinearRegression().fit(log_k, log_lambda)
        slope = reg.coef_[0][0]
        intercept = reg.intercept_[0]
        r2_score = reg.score(log_k, log_lambda)
        estimated_d = 2 / slope if slope > 0.01 else 999.9
        fit_line_y = slope * log_k + intercept
    else:
        estimated_d = 0.0
        slope = 0.0
        r2_score = 0.0
        fit_line_y = None

    k_limit = 20
    eigengaps = np.diff(evals)[:k_limit]

    # 5. DPT & Tau Analysis
    diffusion_evals = 1 - evals
    dc1 = evecs[:, 1]
    sort_i = np.argsort(dc1)
    n_end = max(5, int(len(df_sampled) * 0.05))
    min_c = np.sum(df_sampled.iloc[sort_i[:n_end]]["label"] == "control")
    max_c = np.sum(df_sampled.iloc[sort_i[-n_end:]]["label"] == "control")
    root = sort_i[0] if min_c >= max_c else sort_i[-1]

    target_dims = range(min_dim, max_dim)
    taus = []

    for k in target_dims:
        # 가중치 적용 (Diffusion scale)
        current_weights = diffusion_evals[1 : k + 1] ** t_scale
        diff_coords = evecs[:, 1 : k + 1] * current_weights.reshape(1, -1)

        root_coord = diff_coords[root]
        dpt = np.linalg.norm(diff_coords - root_coord, axis=1)

        if len(nmi_snca_clean) > 10:
            corr, _ = kendalltau(dpt[valid_snca_mask], nmi_snca_clean)
            taus.append(corr)
        else:
            taus.append(0)

    max_tau = np.max(taus)
    avg_tau = np.mean(taus)

    results_summary.append(
        {
            "Metric": name,
            "Est_Dim(d)": estimated_d,
            "R2_Score": r2_score,
            "Max_Tau": max_tau,
            "Avg_Tau": avg_tau,
            "PCA_Applied": "Yes",  # PCA 적용 여부 표시
        }
    )

    # ----------------------------------------------------------------------
    # [시각화]
    # ----------------------------------------------------------------------
    plt.figure(figsize=(18, 5))

    # 제목에 R2 Score도 함께 표시
    plt.suptitle(
        f"Config: {name} | Est. Dim: {estimated_d:.2f} (Slope: {slope:.2f}, $R^2$: {r2_score:.3f})",
        fontsize=14,
        fontweight="bold",
    )

    # Plot 1: Weyl's Law Log-Log Fit
    plt.subplot(1, 3, 1)
    if np.sum(valid_mask) > 3:
        plt.scatter(log_k, log_lambda, color="black", s=20, label="Data")
        # [수정] 라벨에 R^2 추가
        plt.plot(
            log_k,
            fit_line_y,
            "r--",
            label=f"Fit (Slope={slope:.2f}, $R^2$={r2_score:.2f})",
        )
        plt.xlabel("log(k)")
        plt.ylabel("log($\lambda_k$)")
        plt.title(f"Weyl's Law Fit (k={fit_start}~{fit_end})")
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, "Invalid Eigenvalues for Fitting", ha="center")

    # Plot 2: Eigengap
    plt.subplot(1, 3, 2)
    plt.bar(np.arange(1, k_limit + 1), eigengaps, color="skyblue", edgecolor="blue")
    plt.xlabel("Index k")
    plt.ylabel("Gap Size")
    plt.title("Eigengap")
    plt.grid(True, alpha=0.3)

    # Plot 3: Tau Correlation
    plt.subplot(1, 3, 3)
    plt.plot(target_dims, taus, "s-", color="green", linewidth=2)
    plt.axhline(0.15, color="red", linestyle="--")
    plt.xlabel("Target Dimension (k)")
    plt.ylabel("Kendall Tau")
    plt.title(f"Tau (Dim 3-10)\nMax: {max_tau:.3f}")
    plt.xticks(target_dims)
    plt.grid(True, alpha=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# 최종 요약
print("\n" + "=" * 60)
print("      [FINAL SUMMARY]      ")
print("=" * 60)
df_res = pd.DataFrame(results_summary)
if not df_res.empty:
    # 차원(d)과 평균 타우(Avg_Tau)를 함께 출력
    print(
        df_res[["Metric", "Est_Dim(d)", "Max_Tau", "Avg_Tau"]]
        .sort_values(by="Avg_Tau", ascending=False)
        .to_string(index=False)
    )

##### TOP_K = 200
##### t_scale = 0.5
