######## 코랩에서 실행한다고 .ipynb 파일인데 잘 안올라가서 .py로만 바꿔서 올립니다. #######


#### 학습

import os
import glob
import random
import argparse
import logging
import sys
import csv
from typing import List

import numpy as np
import cv2
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.parametrizations import weight_norm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pytorch_metric_learning import losses
from torch.utils.data.dataloader import default_collate
from tqdm.auto import tqdm
import concurrent.futures


# ==============================================================================
# Logging
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ==============================================================================
# 0. User Configuration (HARDCODED FOR COLAB)
# ==============================================================================
DEFAULT_DATA_ROOTS = {
    "Control": [
        "/content/Control_C4/",
        "/content/Control_C18/",
        "/content/Control_C19/",
    ],
    "SNCA": ["/content/SNCA/"],
    "GBA": ["/content/GBA/"],
    "PINK1": ["/content/PINK1/"],
}


# ==============================================================================
# 1. Configuration & Reproducibility
# ==============================================================================
def get_args():
    parser = argparse.ArgumentParser(description="SupCon Training for 16-bit Biological Images")

    # Experiment
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save_dir", type=str, default="/content/drive/MyDrive/Final_paper/Model2", help="Save directory")

    # Data
    parser.add_argument("--max_samples", type=int, default=18000, help="Max samples per class")
    parser.add_argument("--test_ratio", type=float, default=1 / 3, help="Test split ratio")
    parser.add_argument("--val_ratio", type=float, default=0.25, help="Validation split ratio")

    # Training
    parser.add_argument("--img_size", type=int, default=128, help="Input image size")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--temp", type=float, default=0.1, help="Temperature for SupCon loss")
    parser.add_argument("--embed_dim", type=int, default=512, help="Projection head dimension")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")

    # Jupyter/Colab
    if "ipykernel" in sys.modules:
        return parser.parse_args([])
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Speed mode (non-deterministic)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    logger.info(f"Random Seed set to {seed}")
    logger.info(f"cudnn.deterministic={torch.backends.cudnn.deterministic}, cudnn.benchmark={torch.backends.cudnn.benchmark}")


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ==============================================================================
# Data validation & collate
# ==============================================================================
def validate_uint16_rgb_128(img: np.ndarray, filepath: str, img_size: int = 128):
    if img is None:
        raise ValueError("cv2.imread returned None")
    if img.dtype != np.uint16:
        raise ValueError(f"dtype must be uint16, got {img.dtype}")
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"shape must be HxWx3, got {img.shape}")
    h, w = img.shape[:2]
    if (h, w) != (img_size, img_size):
        raise ValueError(f"size must be {(img_size, img_size)}, got {(h, w)}")


def log_skip(filepath: str, reason: Exception):
    print(f"[DATA_SKIP] {filepath} | {type(reason).__name__}: {reason}", file=sys.stderr, flush=True)


def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return default_collate(batch)


# ==============================================================================
# 2. Preprocessing & Dataset
# ==============================================================================
class SafeInstanceNormalize:
    """(x - mu) / max(std, threshold) on CHW tensor."""
    def __init__(self, threshold: float = 0.01):
        self.threshold = float(threshold)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        mean = torch.mean(tensor, dim=[1, 2], keepdim=True)
        std = torch.std(tensor, dim=[1, 2], keepdim=True)
        std = std.clamp_min(self.threshold)
        return (tensor - mean) / std


class InMemorySixteenBitDataset(Dataset):
    """
    - uint16 TIFF (128x128x3) -> float32 [0,1] -> CHW tensor
    - two_crops: return (view1, view2, label)
    - augment: rotation aug ON/OFF
    - preloaded_images: reuse RAM cache
    """
    def __init__(
        self,
        files: List[str],
        labels: List[int],
        img_size: int,
        two_crops: bool,
        augment: bool,
        preloaded_images=None,
    ):
        self.files = files
        self.labels = labels
        self.img_size = img_size
        self.two_crops = two_crops
        self.augment = augment

        if preloaded_images is not None:
            self.images = preloaded_images
        else:
            self.images = [None] * len(self.files)
            print(f"⚡ Loading {len(files)} images into RAM...")

            with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
                list(tqdm(executor.map(self._load_file, range(len(files))), total=len(files), leave=False))

        if self.augment:
            aug = transforms.RandomChoice([
                transforms.Lambda(lambda x: x),
                transforms.Lambda(lambda x: torch.rot90(x, 1, [1, 2])),
                transforms.Lambda(lambda x: torch.rot90(x, 2, [1, 2])),
                transforms.Lambda(lambda x: torch.rot90(x, 3, [1, 2])),
            ])
        else:
            aug = transforms.Lambda(lambda x: x)

        self.transform = transforms.Compose([
            aug,
            SafeInstanceNormalize(threshold=0.01),
        ])

    def _load_file(self, idx: int):
        filepath = self.files[idx]
        try:
            img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
            validate_uint16_rgb_128(img, filepath, self.img_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.images[idx] = img  # keep uint16 in RAM
        except Exception as e:
            log_skip(filepath, e)
            self.images[idx] = None

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        img = self.images[idx]
        if img is None:
            return None

        label = int(self.labels[idx])

        img = img.astype(np.float32) / 65535.0  # HWC float32
        img_tensor = torch.from_numpy(img).permute(2, 0, 1)  # CHW

        if self.two_crops:
            v1 = self.transform(img_tensor)
            v2 = self.transform(img_tensor)
            return v1, v2, torch.tensor(label, dtype=torch.long)
        else:
            x = self.transform(img_tensor)
            return x, torch.tensor(label, dtype=torch.long)


# ==============================================================================
# 3. Data Manager
# ==============================================================================
def save_split_info(files, labels, save_dir, filename):
    path = os.path.join(save_dir, filename)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "label"])
        for fp, lb in zip(files, labels):
            writer.writerow([fp, lb])
    logger.info(f"Saved split info to {path}")


def get_dataloaders(args):
    class_map = {"Control": 0, "SNCA": 1, "GBA": 2, "PINK1": 3}
    all_files, all_labels, stratify_labels = [], [], []
    stratify_counter = 0

    num_control_lines = len(DEFAULT_DATA_ROOTS["Control"])
    target_per_control_line = args.max_samples // num_control_lines

    logger.info("Processing Data Distribution...")

    for class_name, paths in DEFAULT_DATA_ROOTS.items():
        label = class_map[class_name]

        if class_name == "Control":
            for line_idx, line_path in enumerate(paths):
                files = glob.glob(os.path.join(line_path, "**/*.[tT][iI][fF]*"), recursive=True)
                files.sort()
                random.shuffle(files)
                files = files[: min(len(files), target_per_control_line)]

                logger.info(f"  [{class_name} Line {line_idx+1}] Count: {len(files)}")
                all_files.extend(files)
                all_labels.extend([label] * len(files))
                stratify_labels.extend([stratify_counter] * len(files))
                stratify_counter += 1
        else:
            files = []
            for p in paths:
                files.extend(glob.glob(os.path.join(p, "**/*.[tT][iI][fF]*"), recursive=True))
            files.sort()
            random.shuffle(files)
            files = files[: min(len(files), args.max_samples)]

            logger.info(f"  [{class_name}] Count: {len(files)}")
            all_files.extend(files)
            all_labels.extend([label] * len(files))
            stratify_labels.extend([stratify_counter] * len(files))
            stratify_counter += 1

    X_temp, X_test, y_temp, y_test, strat_temp, strat_test = train_test_split(
        all_files, all_labels, stratify_labels,
        test_size=args.test_ratio, random_state=args.seed, stratify=stratify_labels
    )
    X_train, X_val, y_train, y_val, _, _ = train_test_split(
        X_temp, y_temp, strat_temp,
        test_size=args.val_ratio, random_state=args.seed, stratify=strat_temp
    )

    os.makedirs(args.save_dir, exist_ok=True)
    save_split_info(X_train, y_train, args.save_dir, "train_split.csv")
    save_split_info(X_val, y_val, args.save_dir, "val_split.csv")
    save_split_info(X_test, y_test, args.save_dir, "test_split.csv")

    logger.info(f"Split -> Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Train: two crops + aug
    train_ds = InMemorySixteenBitDataset(X_train, y_train, args.img_size, two_crops=True, augment=True)

    # Val: two crops + NO aug (loss monitoring 안정화)
    val_ds = InMemorySixteenBitDataset(
        X_val, y_val, args.img_size, two_crops=True, augment=False
    )

    g = torch.Generator()
    g.manual_seed(args.seed)

    NUM_WORKERS = 0

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
        collate_fn=collate_skip_none,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
        collate_fn=collate_skip_none,
        drop_last=True,
    )

    return train_loader, val_loader


# ==============================================================================
# 4. Model
# ==============================================================================
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(weight_norm(nn.Conv2d(3, 64, 3, 2, 1)), nn.ReLU())
        self.layer2 = nn.Sequential(weight_norm(nn.Conv2d(64, 128, 3, 1, 1)), nn.ReLU())
        self.layer3 = nn.Sequential(weight_norm(nn.Conv2d(128, 256, 3, 1, 2, dilation=2)), nn.ReLU())
        self.layer4 = nn.Sequential(weight_norm(nn.Conv2d(256, 512, 3, 1, 4, dilation=4)), nn.ReLU())
        self.layer5 = nn.Sequential(weight_norm(nn.Conv2d(512, 1024, 3, 1, 2, dilation=2)), nn.ReLU())
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x, return_map=False):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        feat_map = self.layer5(x)
        pooled = self.gap(feat_map).view(feat_map.size(0), -1)
        return (pooled, feat_map) if return_map else pooled


class SupConModel(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.encoder = Encoder()
        self.projector = nn.Sequential(
            weight_norm(nn.Linear(1024, 1024)), nn.ReLU(),
            weight_norm(nn.Linear(1024, embed_dim))
        )

    def forward(self, x):
        pooled = self.encoder(x, return_map=False)
        return F.normalize(self.projector(pooled), dim=1)


# ==============================================================================
# 5. Training
# ==============================================================================
class Trainer:
    def __init__(self, args, model, train_loader, val_loader):
        self.args = args
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.criterion = losses.SupConLoss(temperature=args.temp)
        self.optimizer = optim.AdamW(model.parameters(), lr=args.lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.epochs)
        self.scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.best_loss = float("inf")
        self.patience_counter = 0

    def train_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0.0
        steps = 0

        pbar = tqdm(self.train_loader, desc=f"Train E{epoch}/{self.args.epochs}", leave=True)
        for batch in pbar:
            if batch is None:
                continue
            view1, view2, labels = batch
            if labels.numel() < 2:
                continue

            view1 = view1.to(self.device, non_blocking=True)
            view2 = view2.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            images = torch.cat([view1, view2], dim=0)
            labels2 = torch.cat([labels, labels], dim=0)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=self.device.type, enabled=torch.cuda.is_available()):
                embeddings = self.model(images)
                loss = self.criterion(embeddings, labels2)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            cur = float(loss.item())
            total_loss += cur
            steps += 1
            pbar.set_postfix({"loss": f"{cur:.4f}"})

        return 0.0 if steps == 0 else total_loss / steps

    def validate(self, epoch: int):
        self.model.eval()
        total_loss = 0.0
        steps = 0

        pbar = tqdm(self.val_loader, desc=f"Val   E{epoch}/{self.args.epochs}", leave=True)
        with torch.no_grad():
            for batch in pbar:
                if batch is None:
                    continue
                view1, view2, labels = batch
                if labels.numel() < 2:
                    continue

                view1 = view1.to(self.device, non_blocking=True)
                view2 = view2.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                images = torch.cat([view1, view2], dim=0)
                labels2 = torch.cat([labels, labels], dim=0)

                with torch.amp.autocast(device_type=self.device.type, enabled=torch.cuda.is_available()):
                    embeddings = self.model(images)
                    loss = self.criterion(embeddings, labels2)

                cur = float(loss.item())
                total_loss += cur
                steps += 1
                pbar.set_postfix({"val_loss": f"{cur:.4f}"})

        return 0.0 if steps == 0 else total_loss / steps

    def run(self):
        logger.info(f"Starting Training on {self.device}")
        os.makedirs(self.args.save_dir, exist_ok=True)
        save_path = os.path.join(self.args.save_dir, "best_model.pt")

        for epoch in range(1, self.args.epochs + 1):
            # epoch 시작을 명시적으로 출력 (tqdm에 안 묻히게)
            tqdm.write(f"\n===== Epoch {epoch}/{self.args.epochs} =====")

            train_loss = self.train_epoch(epoch)
            val_loss = self.validate(epoch)

            tqdm.write(f"Epoch {epoch:03d} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.patience_counter = 0
                torch.save(self.model.state_dict(), save_path)
                tqdm.write(f"  -> Saved Best Model (best_val={val_loss:.6f})")
            else:
                self.patience_counter += 1
                tqdm.write(f"  -> No improvement (best_val={self.best_loss:.6f}) [{self.patience_counter}/{self.args.patience}]")
                if self.patience_counter >= self.args.patience:
                    tqdm.write("Early Stopping Triggered")
                    break

            self.scheduler.step()


# ==============================================================================
# Main
# ==============================================================================
if __name__ == "__main__":
    args = get_args()
    set_seed(args.seed)

    train_dl, val_dl = get_dataloaders(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SupConModel(embed_dim=args.embed_dim).to(device)

    trainer = Trainer(args, model, train_dl, val_dl)
    trainer.run()