# ===========================================================
# MULTIMODAL DERM CLASSIFICATION (ViT + Clinical Metadata)
# - Patient-wise splits (no leakage)
# - True cross-attention: metadata tokens attend to ViT patch tokens
# - Multi-token metadata (one token per feature group)
# - Oversample minority class (Benign) via WeightedRandomSampler
# - Label smoothing to stabilize BCE loss (calibration)
# - Train with ViT frozen
# - Early stopping on Val PR-AUC
# - Plot train vs val loss at the end
# ===========================================================

# ===========================================================
# STEP 0) IMPORTS
# ===========================================================
import os, random
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import WeightedRandomSampler

from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchvision import transforms

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score

import matplotlib.pyplot as plt
import json
import shutil

# ===========================================================
# STEP 1) PATHS (EDIT THESE)
# ===========================================================
IMAGES_DIR = "/content/drive/../images"
METADATA_CSV = "/content/drive/../metadata.csv"
OUTDIR = "/content/drive/../results"
os.makedirs(OUTDIR, exist_ok=True)

# ===========================================================
# STEP 2) REPRODUCIBILITY + DEVICE
# ===========================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN_MEMORY = (DEVICE.type == "cuda")
print(f"Using device: {DEVICE}")

# ===========================================================
# STEP 3) LOAD METADATA + BASIC FILTERING
# ===========================================================
df = pd.read_csv(METADATA_CSV)

df = df[df["diagnosis_1"].isin(["Malignant", "Benign"])].copy()
df["target"] = (df["diagnosis_1"] == "Malignant").astype(int)
df["img_path"] = df["isic_id"].apply(lambda x: os.path.join(IMAGES_DIR, f"{x}.jpg"))

df = df[df["img_path"].apply(os.path.exists)].reset_index(drop=True)

print(f"Total usable samples: {len(df)}")
print("Class distribution:")
print(df["target"].value_counts())

# ===========================================================
# STEP 4) PATIENT-WISE SPLITS (TRAIN / VAL / TEST)
# ===========================================================
gss_test = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
train_full_idx, test_idx = next(gss_test.split(df, df["target"], groups=df["patient_id"]))
train_full = df.iloc[train_full_idx].reset_index(drop=True)
test_df    = df.iloc[test_idx].reset_index(drop=True)

gss_val = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
train_idx, val_idx = next(gss_val.split(train_full, train_full["target"], groups=train_full["patient_id"]))
train_df = train_full.iloc[train_idx].reset_index(drop=True)
val_df   = train_full.iloc[val_idx].reset_index(drop=True)

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# Sanity: no patient overlap
train_pat = set(train_df["patient_id"].astype(str).unique())
val_pat   = set(val_df["patient_id"].astype(str).unique())
test_pat  = set(test_df["patient_id"].astype(str).unique())
print(f"Patient overlap train/val:  {len(train_pat & val_pat)}")
print(f"Patient overlap train/test: {len(train_pat & test_pat)}")
print(f"Patient overlap val/test:   {len(val_pat & test_pat)}")

# ===========================================================
# STEP 5) METADATA PREPROCESSING (NUM + CAT)
# ===========================================================
NUM_COLS = ["age_approx", "clin_size_long_diam_mm"]
CAT_COLS = ["sex", "fitzpatrick_skin_type", "anatom_site_general"]

# Ensure columns exist
for c in NUM_COLS + CAT_COLS:
    for dfx in (train_df, val_df, test_df):
        if c not in dfx.columns:
            dfx[c] = np.nan

def prep_metadata(dfx: pd.DataFrame) -> pd.DataFrame:
    dfx = dfx.copy()
    for c in CAT_COLS:
        dfx[c] = dfx[c].fillna("Unknown").astype(str)
    for c in NUM_COLS:
        dfx[c] = pd.to_numeric(dfx[c], errors="coerce")
    return dfx

train_df = prep_metadata(train_df)
val_df   = prep_metadata(val_df)
test_df  = prep_metadata(test_df)

# Numeric normalization (train-only)
num_medians = train_df[NUM_COLS].median()
num_means   = train_df[NUM_COLS].fillna(num_medians).mean()
num_stds    = train_df[NUM_COLS].fillna(num_medians).std().replace(0, 1)

def norm_num_and_mask(dfx: pd.DataFrame):
    """
    Returns:
      x_num: [N, n_num] normalized
      x_miss: [N, n_num] missingness mask (1 if missing, else 0)
    """
    raw = dfx[NUM_COLS]
    miss = raw.isna().values.astype(np.float32)
    x = raw.fillna(num_medians)
    x = (x - num_means) / num_stds
    return x.values.astype(np.float32), miss

# Categorical vocab (train-only)
cat_vocabs = {}
for c in CAT_COLS:
    uniq = sorted(train_df[c].unique().tolist())
    if "Unknown" not in uniq:
        uniq = ["Unknown"] + uniq
    cat_vocabs[c] = {v: i for i, v in enumerate(uniq)}

def encode_cat(dfx: pd.DataFrame) -> np.ndarray:
    mats = []
    for c in CAT_COLS:
        vocab = cat_vocabs[c]
        mats.append(dfx[c].map(lambda v: vocab.get(v, vocab["Unknown"])).values.astype(np.int64))
    return np.stack(mats, axis=1)  # [N, n_cat]

# ===========================================================
# STEP 6) IMAGE TRANSFORMS
# ===========================================================
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.10, hue=0.02),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ===========================================================
# STEP 7) DATASET
# ===========================================================
class ISICMultimodalDataset(Dataset):
    def __init__(self, dfx: pd.DataFrame, transform=None):
        self.dfx = dfx.reset_index(drop=True)
        self.y = self.dfx["target"].values.astype(np.float32)

        self.num, self.num_miss = norm_num_and_mask(self.dfx)
        self.cat = encode_cat(self.dfx)
        self.transform = transform

    def __len__(self):
        return len(self.dfx)

    def __getitem__(self, idx):
        path = self.dfx.loc[idx, "img_path"]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), color="gray")

        img = self.transform(img) if self.transform else val_transform(img)

        x_num = torch.tensor(self.num[idx], dtype=torch.float32)
        x_miss = torch.tensor(self.num_miss[idx], dtype=torch.float32)
        x_cat = torch.tensor(self.cat[idx], dtype=torch.long)
        y = torch.tensor(self.y[idx], dtype=torch.float32)

        return img, x_num, x_miss, x_cat, y

# ===========================================================
# STEP 8) MODEL: MULTI-TOKEN METADATA + CROSS-ATTENTION
# ===========================================================
class MetadataTokenizer(nn.Module):
    """
    Produces a sequence of metadata tokens:
      - one token per categorical feature (embedding -> projected to hidden)
      - one token for numeric features (+ missingness mask) -> projected to hidden
    Output: [B, T_meta, hidden]
    """
    def __init__(self, cat_sizes, num_dim, hidden=768, cat_emb_dim=64):
        super().__init__()
        self.hidden = hidden
        self.num_dim = num_dim
        self.n_cat = len(cat_sizes)

        self.cat_embs = nn.ModuleList([nn.Embedding(sz, cat_emb_dim) for sz in cat_sizes])
        self.cat_proj = nn.ModuleList([
            nn.Sequential(nn.Linear(cat_emb_dim, hidden), nn.GELU(), nn.LayerNorm(hidden))
            for _ in cat_sizes
        ])

        self.num_proj = nn.Sequential(
            nn.Linear(num_dim * 2, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden)
        )

    def forward(self, x_num, x_miss, x_cat):
        cat_tokens = []
        for i, (emb, proj) in enumerate(zip(self.cat_embs, self.cat_proj)):
            t = emb(x_cat[:, i])
            t = proj(t)
            cat_tokens.append(t.unsqueeze(1))

        num_in = torch.cat([x_num, x_miss], dim=1)
        num_tok = self.num_proj(num_in).unsqueeze(1)

        tokens = torch.cat(cat_tokens + [num_tok], dim=1)
        return tokens


class CrossAttnViT_MultiMeta(nn.Module):
    """
    - Extract ViT tokens: [B, 197, 768] (CLS + patches)
    - Metadata tokens query image tokens via cross-attention (meta->image)
    - Pool metadata tokens + CLS token for classification
    """
    def __init__(self, cat_sizes, num_dim, hidden=768, nheads=8, dropout=0.3):
        super().__init__()
        self.vit = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        self.vit.heads = nn.Identity()

        self.meta_tok = MetadataTokenizer(cat_sizes, num_dim, hidden=hidden, cat_emb_dim=64)

        self.type_emb = nn.Embedding(2, hidden)

        self.cross_attn = nn.MultiheadAttention(hidden, nheads, dropout=dropout, batch_first=True)
        self.cross_ln = nn.LayerNorm(hidden)

        self.cross_ff = nn.Sequential(
            nn.Linear(hidden, hidden * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 4, hidden),
            nn.Dropout(dropout),
        )
        self.cross_ff_ln = nn.LayerNorm(hidden)

        self.classifier = nn.Sequential(
            nn.Linear(hidden * 2, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

    def _vit_tokens(self, x):
        x = self.vit._process_input(x)
        b = x.shape[0]
        cls = self.vit.class_token.expand(b, -1, -1)
        x = torch.cat((cls, x), dim=1)

        x = x + self.vit.encoder.pos_embedding
        x = self.vit.encoder.dropout(x)
        x = self.vit.encoder.layers(x)
        x = self.vit.encoder.ln(x)
        return x

    def forward(self, img, x_num, x_miss, x_cat):
        img_tokens = self._vit_tokens(img)
        cls_tok = img_tokens[:, 0, :]

        meta_tokens = self.meta_tok(x_num, x_miss, x_cat)

        img_tokens = img_tokens + self.type_emb(
            torch.zeros(img_tokens.size(1), device=img_tokens.device, dtype=torch.long)
        ).unsqueeze(0)
        meta_tokens = meta_tokens + self.type_emb(
            torch.ones(meta_tokens.size(1), device=meta_tokens.device, dtype=torch.long)
        ).unsqueeze(0)

        attn_out, _ = self.cross_attn(query=meta_tokens, key=img_tokens, value=img_tokens)
        meta_upd = self.cross_ln(meta_tokens + attn_out)

        meta_ff = self.cross_ff_ln(meta_upd + self.cross_ff(meta_upd))
        meta_pooled = meta_ff.mean(dim=1)

        fused = torch.cat([cls_tok, meta_pooled], dim=1)
        logits = self.classifier(fused)
        return logits

# ===========================================================
# STEP 9) FREEZE BACKBONE (ViT)
# ===========================================================
def set_requires_grad(module: nn.Module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag

def freeze_vit_all(model: nn.Module):
    set_requires_grad(model.vit, False)

# ===========================================================
# STEP 10) OPTIMIZER (ONLY TRAIN HEAD/FUSION WHEN VIT IS FROZEN)
# ===========================================================
def build_optimizer(model: nn.Module, lr=3e-4, wd=1e-4):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=lr, weight_decay=wd)

# ===========================================================
# STEP 11) EVALUATION
# ===========================================================
def evaluate(model, loader, device):
    model.eval()
    losses, probs, labels = [], [], []

    with torch.no_grad():
        for img, x_num, x_miss, x_cat, y in loader:
            img = img.to(device)
            x_num = x_num.to(device)
            x_miss = x_miss.to(device)
            x_cat = x_cat.to(device)
            y = y.to(device).unsqueeze(1)

            logits = model(img, x_num, x_miss, x_cat)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            losses.append(loss.item())

            p = torch.sigmoid(logits).detach().cpu().numpy().ravel()
            probs.append(p)
            labels.append(y.detach().cpu().numpy().ravel())

    probs = np.concatenate(probs)
    labels = np.concatenate(labels).astype(int)
    roc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else float("nan")
    pr  = average_precision_score(labels, probs) if len(np.unique(labels)) > 1 else float("nan")
    return float(np.mean(losses)), float(roc), float(pr), probs, labels

# ===========================================================
# STEP 12) TRAINING LOOP (LABEL SMOOTHING + EARLY STOP ON VAL PR-AUC)
# ===========================================================
def train_model(
    model,
    train_loader,
    val_loader,
    device,
    epochs=30,
    patience=8,
    lr=3e-4,
    wd=1e-4,
    label_smoothing=0.05,
    outdir=OUTDIR,
):
    optimizer = build_optimizer(model, lr=lr, wd=wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, threshold=1e-4
    )

    best_val_pr = -1.0
    best_epoch = 0
    patience_ctr = 0

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_roc": [],
        "train_pr": [],
        "val_roc": [],
        "val_pr": [],
        "lr": [],
    }

    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    for epoch in range(1, epochs + 1):
        model.train()
        tr_losses, tr_probs, tr_labels = [], [], []

        for img, x_num, x_miss, x_cat, y in train_loader:
            img = img.to(device)
            x_num = x_num.to(device)
            x_miss = x_miss.to(device)
            x_cat = x_cat.to(device)
            y = y.to(device).unsqueeze(1)

            eps = float(label_smoothing)
            y_smooth = y * (1 - eps) + 0.5 * eps

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(img, x_num, x_miss, x_cat)
                loss = F.binary_cross_entropy_with_logits(logits, y_smooth)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            tr_losses.append(loss.item())
            tr_probs.append(torch.sigmoid(logits).detach().cpu().numpy().ravel())
            tr_labels.append(y.detach().cpu().numpy().ravel())

        tr_probs = np.concatenate(tr_probs)
        tr_labels = np.concatenate(tr_labels).astype(int)

        tr_loss = float(np.mean(tr_losses))
        tr_roc = roc_auc_score(tr_labels, tr_probs) if len(np.unique(tr_labels)) > 1 else float("nan")
        tr_pr  = average_precision_score(tr_labels, tr_probs) if len(np.unique(tr_labels)) > 1 else float("nan")

        val_loss, val_roc, val_pr, _, _ = evaluate(model, val_loader, device)

        scheduler.step(-val_pr)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"Train loss {tr_loss:.4f} ROC {tr_roc:.4f} PR {tr_pr:.4f} | "
            f"Val loss {val_loss:.4f} ROC {val_roc:.4f} PR {val_pr:.4f} | "
            f"LR {current_lr:.2e}"
        )

        history["epoch"].append(epoch)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_roc"].append(tr_roc)
        history["train_pr"].append(tr_pr)
        history["val_roc"].append(val_roc)
        history["val_pr"].append(val_pr)
        history["lr"].append(current_lr)

        if val_pr > best_val_pr + 1e-4:
            best_val_pr = val_pr
            best_epoch = epoch
            patience_ctr = 0
            torch.save(model.state_dict(), os.path.join(outdir, "best_model.pth"))
            print(f"  ⭐ New best model by Val PR-AUC: {best_val_pr:.4f}")
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"✅ Early stopping at epoch {epoch}. Best epoch {best_epoch}, best Val PR-AUC {best_val_pr:.4f}")
                break

    return history

# ===========================================================
# STEP 13) CHECK DISK
# ===========================================================
total, used, free = shutil.disk_usage(OUTDIR)
print(f"Disk (OUTDIR): total {total // (2**30)} GB | used {used // (2**30)} GB | free {free // (2**30)} GB")

# ===========================================================
# STEP 14) BUILD DATASETS + DATALOADERS
# ===========================================================
cat_sizes = [len(cat_vocabs[c]) for c in CAT_COLS]

train_dataset = ISICMultimodalDataset(train_df, transform=train_transform)
val_dataset   = ISICMultimodalDataset(val_df, transform=val_transform)
test_dataset  = ISICMultimodalDataset(test_df, transform=val_transform)

print(f"Dataset sizes -> Train {len(train_dataset)} | Val {len(val_dataset)} | Test {len(test_dataset)}")

y_train = train_dataset.y.astype(int)
class_counts = np.bincount(y_train)
class_weights = 1.0 / np.maximum(class_counts, 1)
sample_weights = class_weights[y_train]

sampler = WeightedRandomSampler(
    weights=torch.DoubleTensor(sample_weights),
    num_samples=len(sample_weights),
    replacement=True
)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    sampler=sampler,
    shuffle=False,
    num_workers=2,
    pin_memory=PIN_MEMORY
)
val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=2,
    pin_memory=PIN_MEMORY
)
test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=2,
    pin_memory=PIN_MEMORY
)

# ===========================================================
# STEP 15) INIT MODEL + FREEZE ViT
# ===========================================================
model = CrossAttnViT_MultiMeta(cat_sizes, num_dim=len(NUM_COLS), dropout=0.3).to(DEVICE)
freeze_vit_all(model)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total model parameters: {total_params:,}")
print(f"Trainable parameters (ViT frozen): {trainable_params:,}")

# ===========================================================
# STEP 16) TRAIN (ViT frozen only)
# ===========================================================
history = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device=DEVICE,
    epochs=30,
    patience=8,
    lr=3e-4,
    wd=1e-4,
    label_smoothing=0.05,
    outdir=OUTDIR
)

# ===========================================================
# STEP 17) FINAL EVALUATION ON TEST SET (LOAD BEST)
# ===========================================================
best_path = os.path.join(OUTDIR, "best_model.pth")
if os.path.exists(best_path):
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    print("Loaded best_model.pth (best by Val PR-AUC)")

test_loss, test_roc, test_pr, test_probs, y_test = evaluate(model, test_loader, DEVICE)
print(f"Test -> loss {test_loss:.4f} | ROC-AUC {test_roc:.4f} | PR-AUC {test_pr:.4f}")

np.save(os.path.join(OUTDIR, "model4_probs.npy"), test_probs)
np.save(os.path.join(OUTDIR, "y_test.npy"), y_test)

print("✅ Saved model4_probs.npy and y_test.npy")

# ===========================================================
# STEP 18) SAVE RESULTS (JSON)
# ===========================================================
results = {
    "seed": SEED,
    "device": str(DEVICE),
    "samples": {
        "total": int(len(df)),
        "train": int(len(train_df)),
        "val": int(len(val_df)),
        "test": int(len(test_df))
    },
    "class_distribution_total": df["target"].value_counts().to_dict(),
    "train_class_counts": {
        "benign_0": int(class_counts[0]),
        "malignant_1": int(class_counts[1])
    },
    "model": {
        "backbone": "vit_b_16 (frozen)",
        "fusion": "cross_attention(meta_tokens->vit_tokens)",
        "meta_tokens": f"{len(CAT_COLS)} categorical + 1 numeric",
        "label_smoothing": 0.05
    },
    "best_val_pr_auc": float(np.max(history["val_pr"])) if len(history["val_pr"]) else None,
    "test_metrics": {
        "loss": float(test_loss),
        "roc_auc": float(test_roc),
        "pr_auc": float(test_pr)
    },
    "saved_outputs": {
        "best_model": os.path.join(OUTDIR, "best_model.pth"),
        "test_probabilities": os.path.join(OUTDIR, "model4_probs.npy"),
        "test_labels": os.path.join(OUTDIR, "y_test.npy")
    },
    "history": history
}

with open(os.path.join(OUTDIR, "multimodal_crossattn_results.json"), "w") as f:
    json.dump(results, f, indent=2)

print(f"✅ Saved: {best_path}")
print(f"✅ Saved: {os.path.join(OUTDIR, 'multimodal_crossattn_results.json')}")

# ===========================================================
# STEP 19) PLOT TRAIN LOSS VS VAL LOSS
# ===========================================================
if len(history["epoch"]) > 0:
    plt.figure()
    plt.plot(history["epoch"], history["train_loss"], label="Train Loss")
    plt.plot(history["epoch"], history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train vs Validation Loss Over Epochs")
    plt.legend()
    plt.tight_layout()
    plt.show()
