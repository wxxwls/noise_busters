

import os, json, math, random, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score, confusion_matrix

try:
    import timm
except:
    raise SystemExit("timm가 필요합니다. 먼저 `pip install timm` 실행하세요.")

# ---------------- CONFIG ----------------
CFG = {
    "DATA_ROOT": "/content/drive/MyDrive/kaggle_noise",  # ★ 전처리와 동일
    "CSV_PREFIX": "ast_split_with_spec",
    "SEED": 42,
    "BATCH": 32,
    "EPOCHS": 60,
    "LR": 2e-4,
    "WEIGHT_DECAY": 1e-4,
    "WARMUP_EPOCHS": 4,
    "HUBER_DELTA": 1.5,
    "CLS_SMOOTH": 0.05,
    "EMA_DECAY": 0.999,
    "EARLY_STOP": 10,
    "MIXED_PREC": True,
    # SpecAug (기본 off)
    "SPEC_FREQ_MASKS": 2, "SPEC_FREQ_MAXW": 16,
    "SPEC_TIME_MASKS": 2, "SPEC_TIME_MAXW": 24,
    "SPEC_AUG_PROB": 0.0,
    # 로더
    "NUM_WORKERS": 2, "PIN_MEMORY": True, "PERSISTENT": True, "PREFETCH": 2,
    # 스코어
    "SCORE_ALPHA": 1.0,
    "SAVE_DIR": "convnext_tiny_multi_ckpt",
    # 시간축 시프트 (train만)
    "TIME_SHIFT_MAX": 24,
    # 검증/테스트 TTA
    "TTA_N": 1, "TTA_MAX_SHIFT": 0,
    # 메타 피처
    "USE_META": True,
    "META_COLS": ["floor", "intensity", "distance", "repeat"],
    "META_EMB_DIM": 64,
}

RESIZE_HW = (224, 224)

# ---------------- Utils ----------------
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if torch.__version__ >= "2.0":
        try: torch.set_float32_matmul_precision("high")
        except: pass

def label_smoothing_ce(logits, target, smoothing=0.0):
    if smoothing <= 0: return F.cross_entropy(logits, target)
    n = logits.size(-1)
    with torch.no_grad():
        true = torch.zeros_like(logits).fill_(smoothing / (n - 1))
        true.scatter_(1, target.unsqueeze(1), 1.0 - smoothing)
    logp = F.log_softmax(logits, dim=1)
    return -(true * logp).sum(dim=1).mean()

def spec_augment(x, p=0.7, f_masks=2, f_w=16, t_masks=2, t_w=24):
    if p <= 0 or random.random() > p: return x
    B, C, Fm, Tm = x.shape
    x = x.clone()
    for b in range(B):
        for _ in range(f_masks):
            w = random.randint(0, max(0, f_w))
            if w == 0: continue
            f0 = random.randint(0, max(0, Fm - w))
            x[b, :, f0:f0+w, :] = 0.0
        for _ in range(t_masks):
            w = random.randint(0, max(0, t_w))
            if w == 0: continue
            t0 = random.randint(0, max(0, Tm - w))
            x[b, :, :, t0:t0+w] = 0.0
    return x

def per_image_standardize(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 3:
        mean = x.mean(dim=(1, 2), keepdim=True)
        std  = x.std(dim=(1, 2), keepdim=True).clamp_min(1e-5)
        return (x - mean) / std
    elif x.dim() == 4:
        mean = x.mean(dim=(2, 3), keepdim=True)
        std  = x.std(dim=(2, 3), keepdim=True).clamp_min(1e-5)
        return (x - mean) / std
    else:
        raise ValueError(f"per_image_standardize: unexpected shape {tuple(x.shape)}")

def time_shift(x, max_shift):
    if max_shift <= 0: return x
    s = random.randint(-max_shift, max_shift)
    return torch.roll(x, shifts=s, dims=3)

# ---------------- EMA ----------------
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {n: p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
        self.backup = {}

    @torch.no_grad()
    def update(self, model):
        for n, p in model.named_parameters():
            if not p.requires_grad: continue
            self.shadow[n] = self.decay*self.shadow[n] + (1.0-self.decay)*p.detach()

    def apply_shadow(self, model):
        self.backup = {}
        for n, p in model.named_parameters():
            if not p.requires_grad: continue
            self.backup[n] = p.detach().clone()
            p.data.copy_(self.shadow[n].data)

    def restore(self, model):
        for n, p in model.named_parameters():
            if not p.requires_grad: continue
            p.data.copy_(self.backup[n].data)
        self.backup = {}

# ---------------- Dataset ----------------
class LogMelDataset(Dataset):
    def __init__(self, csv_path, le=None, meta_cols=None, scaler=None, fit_scaler=False):
        self.df = pd.read_csv(csv_path)
        assert {"spec_path","db","cls"}.issubset(self.df.columns)
        self.paths = self.df["spec_path"].tolist()
        self.db = self.df["db"].astype(np.float32).values
        self.cls_raw = self.df["cls"].astype(str).values

        if le is None:
            self.le = LabelEncoder()
            self.cls = self.le.fit_transform(self.cls_raw)
        else:
            self.le = le
            self.cls = self.le.transform(self.cls_raw)

        self.meta_cols = meta_cols or []
        self.has_meta = len(self.meta_cols) > 0 and set(self.meta_cols).issubset(self.df.columns)
        self.scaler = scaler
        if self.has_meta:
            Xm = self.df[self.meta_cols].copy()
            for c in self.meta_cols:
                Xm[c] = pd.to_numeric(Xm[c], errors="coerce")
            Xm = Xm.fillna(Xm.mean())
            if fit_scaler:
                self.scaler = StandardScaler()
                self.scaler.fit(Xm.values.astype(np.float32))
            self.meta_np = self.scaler.transform(Xm.values.astype(np.float32)) if self.scaler is not None else Xm.values.astype(np.float32)
        else:
            self.meta_np = None

    def __len__(self): return len(self.paths)

    def __getitem__(self, i):
        X = np.load(self.paths[i], mmap_mode="r")
        X = np.array(X, dtype=np.float32, copy=True)  # [F,T]
        x = torch.from_numpy(X).unsqueeze(0)          # [1,F,T]
        x = F.interpolate(x.unsqueeze(0), size=RESIZE_HW,
                          mode="bilinear", align_corners=False).squeeze(0)  # [1,224,224]
        x = per_image_standardize(x)
        y_reg = torch.tensor(self.db[i], dtype=torch.float32)
        y_cls = torch.tensor(self.cls[i], dtype=torch.long)

        if self.has_meta:
            m = torch.tensor(self.meta_np[i], dtype=torch.float32)  # [M]
            return x, m, y_reg, y_cls
        else:
            return x, None, y_reg, y_cls

# ---------------- Model ----------------
class ConvNeXtMultiTaskMeta(nn.Module):
    def __init__(self, num_classes, meta_dim=0, meta_emb=64):
        super().__init__()
        self.backbone = timm.create_model("convnext_tiny", pretrained=True, in_chans=1, num_classes=0)
        feat_dim = self.backbone.num_features  # 768
        self.use_meta = meta_dim > 0
        if self.use_meta:
            self.meta_mlp = nn.Sequential(
                nn.Linear(meta_dim, meta_emb), nn.GELU(),
                nn.LayerNorm(meta_emb),
            )
            head_in = feat_dim + meta_emb
        else:
            self.meta_mlp = None
            head_in = feat_dim

        self.head_reg = nn.Sequential(
            nn.LayerNorm(head_in), nn.Linear(head_in, head_in//2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(head_in//2, 1)
        )
        self.head_cls = nn.Sequential(
            nn.LayerNorm(head_in), nn.Linear(head_in, head_in//2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(head_in//2, num_classes)
        )

    def forward(self, x, m=None):
        feat = self.backbone(x)  # [B,C]
        if self.use_meta and m is not None:
            m_emb = self.meta_mlp(m)  # [B,meta_emb]
            feat = torch.cat([feat, m_emb], dim=1)
        y_reg = self.head_reg(feat).squeeze(1)
        y_cls = self.head_cls(feat)
        return y_reg, y_cls

# ---------------- Sched / Loader ----------------
def mae(x, y): return torch.mean(torch.abs(x - y))

def cosine_with_warmup(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            return float(current_epoch + 1) / float(max(1, warmup_epochs))
        progress = (current_epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def make_loader(ds, bs, shuffle):
    return DataLoader(
        ds, batch_size=bs, shuffle=shuffle,
        num_workers=CFG["NUM_WORKERS"], pin_memory=CFG["PIN_MEMORY"],
        persistent_workers=CFG["PERSISTENT"], prefetch_factor=CFG["PREFETCH"]
    )

# ---------------- TTA ----------------
def tta_time_roll(x: torch.Tensor, n: int, max_shift: int):
    if n <= 1 or max_shift <= 0: return [x]
    outs = [x]
    for _ in range(n-1):
        s = random.randint(-max_shift, max_shift)
        outs.append(torch.roll(x, shifts=s, dims=3))
    return outs

# ---------------- Plot helpers ----------------
def plot_curve(xs, ys, title, xlabel, ylabel, save_path):
    plt.figure()
    plt.plot(xs, ys)
    plt.title(title); plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()

def plot_mae_dual(hist_df, save_path):
    """요청 그래프: train/val MAE 함께 그리고, best val에 세로 점선."""
    xs = hist_df["epoch"].values
    y_tr = hist_df["tr_reg_mae"].values
    y_va = hist_df["va_mae"].values
    best_i = int(np.nanargmin(y_va))
    best_epoch = int(xs[best_i])
    best_val = float(y_va[best_i])

    plt.figure()
    plt.plot(xs, y_tr, label="train")
    plt.plot(xs, y_va, label="val")
    plt.axvline(best_epoch, linestyle="--")  # 세로 점선
    plt.title(f"MAE (best@{best_epoch}, val={best_val:.3f})")
    plt.xlabel("epoch"); plt.ylabel("MAE(dB)")
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()

def plot_scatter(x, y, title, xlabel, ylabel, save_path, refline=True):
    plt.figure()
    plt.scatter(x, y, s=10, alpha=0.6)
    if refline:
        lo = min(np.min(x), np.min(y)); hi = max(np.max(x), np.max(y))
        plt.plot([lo, hi], [lo, hi])
    plt.title(title); plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()

def plot_hist(vals, title, xlabel, save_path, bins=30):
    plt.figure()
    plt.hist(vals, bins=bins)
    plt.title(title); plt.xlabel(xlabel); plt.ylabel("freq")
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()

def plot_confmat(y_true, y_pred, label_map, save_path, normalize=True):
    labels = [label_map[i] for i in range(len(label_map))]
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    if normalize:
        cm = cm.astype(np.float32)
        cm = cm / (cm.sum(axis=1, keepdims=True) + 1e-9)
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix"); plt.xlabel("Predicted"); plt.ylabel("True")
    plt.xticks(ticks=np.arange(len(labels)), labels=labels, rotation=45, ha="right")
    plt.yticks(ticks=np.arange(len(labels)), labels=labels)
    plt.colorbar()
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()

# ---------------- Train/Eval ----------------
def train_one_epoch(model, loader, optim, amp_scaler, device, ema=None):
    model.train()
    tot, loss_sum, mae_sum, cls_sum = 0, 0.0, 0.0, 0.0
    for batch in loader:
        if CFG["USE_META"]:
            x, m, y_reg, y_cls = batch
            m = m.to(device, non_blocking=True)
        else:
            x, _, y_reg, y_cls = batch
            m = None
        x = x.to(device, non_blocking=True)
        y_reg = y_reg.to(device, non_blocking=True)
        y_cls = y_cls.to(device, non_blocking=True)

        x = time_shift(x, CFG["TIME_SHIFT_MAX"])
        if CFG["SPEC_AUG_PROB"] > 0:
            x = spec_augment(x, p=CFG["SPEC_AUG_PROB"],
                             f_masks=CFG["SPEC_FREQ_MASKS"], f_w=CFG["SPEC_FREQ_MAXW"],
                             t_masks=CFG["SPEC_TIME_MASKS"], t_w=CFG["SPEC_TIME_MAXW"])

        optim.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type='cuda', enabled=CFG["MIXED_PREC"] and torch.cuda.is_available()):
            pr_reg, pr_cls = model(x, m)
            loss_reg = F.huber_loss(pr_reg, y_reg, delta=CFG["HUBER_DELTA"])
            loss_cls = label_smoothing_ce(pr_cls, y_cls, smoothing=CFG["CLS_SMOOTH"])
            loss = loss_reg + loss_cls

        if amp_scaler is not None:
            amp_scaler.scale(loss).backward()
            amp_scaler.unscale_(optim)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            amp_scaler.step(optim); amp_scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

        if ema is not None: ema.update(model)

        bsz = x.size(0); tot += bsz
        loss_sum += loss.item() * bsz
        cls_sum  += loss_cls.item() * bsz
        with torch.no_grad(): mae_sum += mae(pr_reg, y_reg).item() * bsz

    return {"loss": loss_sum/tot, "reg_mae": mae_sum/tot, "cls_loss": cls_sum/tot}

@torch.no_grad()
def evaluate(model, loader, device, le, use_ema=None, collect=False):
    model.eval()
    if use_ema is not None: use_ema.apply_shadow(model)

    tot = 0
    loss_sum = 0.0; mae_sum = 0.0; cls_sum = 0.0
    y_true = []; y_pred = []
    y_reg_true = []; y_reg_pred = []

    for batch in loader:
        if CFG["USE_META"]:
            x, m, y_reg, y_cls = batch
            m = m.to(device, non_blocking=True)
        else:
            x, _, y_reg, y_cls = batch
            m = None
        x = x.to(device, non_blocking=True)
        y_reg = y_reg.to(device, non_blocking=True)
        y_cls = y_cls.to(device, non_blocking=True)

        x_list = tta_time_roll(x, CFG["TTA_N"], CFG["TTA_MAX_SHIFT"])
        pr_reg_list = []; pr_cls_list = []
        for xt in x_list:
            pr_reg_t, pr_cls_t = model(xt, m)
            pr_reg_list.append(pr_reg_t); pr_cls_list.append(pr_cls_t)
        pr_reg = torch.stack(pr_reg_list, dim=0).mean(0)
        pr_cls = torch.stack(pr_cls_list, dim=0).mean(0)

        loss_reg = F.huber_loss(pr_reg, y_reg, delta=CFG["HUBER_DELTA"])
        loss_cls = F.cross_entropy(pr_cls, y_cls)
        loss = loss_reg + loss_cls

        bsz = x.size(0); tot += bsz
        loss_sum += loss.item() * bsz
        cls_sum  += loss_cls.item() * bsz
        mae_sum  += torch.mean(torch.abs(pr_reg - y_reg)).item() * bsz

        y_true.append(y_cls.cpu().numpy())
        y_pred.append(pr_cls.argmax(1).cpu().numpy())
        y_reg_true.append(y_reg.cpu().numpy())
        y_reg_pred.append(pr_reg.cpu().numpy())

    if use_ema is not None: use_ema.restore(model)

    y_true = np.concatenate(y_true); y_pred = np.concatenate(y_pred)
    y_reg_true = np.concatenate(y_reg_true); y_reg_pred = np.concatenate(y_reg_pred)

    f1_macro = float(f1_score(y_true, y_pred, average="macro"))
    acc = float((y_true == y_pred).mean())
    diff = y_reg_true - y_reg_pred
    mae_val = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))

    out = {
        "loss": loss_sum / tot, "reg_mae": mae_sum / tot, "cls_loss": cls_sum / tot,
        "f1_macro": f1_macro, "acc": acc, "mae": mae_val, "rmse": rmse,
        "label_map": dict(enumerate(le.classes_.tolist()))
    }
    if collect:
        out.update({"y_true": y_true, "y_pred": y_pred,
                    "y_reg_true": y_reg_true, "y_reg_pred": y_reg_pred})
    return out

def composite_score(mae_val, f1_macro, alpha=1.0):
    return mae_val - alpha * f1_macro

# ---------------- Main ----------------
def main():
    set_seed(CFG["SEED"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path(CFG["DATA_ROOT"])
    save_dir = Path(CFG["SAVE_DIR"]); save_dir.mkdir(parents=True, exist_ok=True)

    train_csv = data_root / f"{CFG['CSV_PREFIX']}_train.csv"
    val_csv   = data_root / f"{CFG['CSV_PREFIX']}_val.csv"
    test_csv  = data_root / f"{CFG['CSV_PREFIX']}_test.csv"
    assert train_csv.exists() and val_csv.exists() and test_csv.exists(), "전처리 CSV가 없습니다. preprocess 먼저!"

    tmp = pd.read_csv(train_csv)
    le = LabelEncoder().fit(tmp["cls"].astype(str).values)
    num_classes = len(le.classes_)
    print("[classes]", le.classes_)

    # 메타 스케일러 적합
    meta_cols = CFG["META_COLS"] if CFG["USE_META"] else []
    meta_scaler = None
    if CFG["USE_META"]:
        Xm = tmp[meta_cols].copy()
        for c in meta_cols:
            Xm[c] = pd.to_numeric(Xm[c], errors="coerce")
        meta_scaler = StandardScaler().fit(Xm.values.astype(np.float32))

    # 데이터셋/로더
    train_ds = LogMelDataset(train_csv, le=le, meta_cols=meta_cols, scaler=meta_scaler, fit_scaler=False)
    val_ds   = LogMelDataset(val_csv,   le=le, meta_cols=meta_cols, scaler=meta_scaler, fit_scaler=False)
    test_ds  = LogMelDataset(test_csv,  le=le, meta_cols=meta_cols, scaler=meta_scaler, fit_scaler=False)

    train_loader = make_loader(train_ds, CFG["BATCH"], True)
    val_loader   = make_loader(val_ds,   CFG["BATCH"], False)
    test_loader  = make_loader(test_ds,  CFG["BATCH"], False)

    meta_dim = len(meta_cols) if CFG["USE_META"] else 0
    model = ConvNeXtMultiTaskMeta(num_classes=num_classes, meta_dim=meta_dim, meta_emb=CFG["META_EMB_DIM"]).to(device)

    # 회귀 바이어스 = 학습셋 평균 dB
    train_mean = float(tmp["db"].astype(float).mean())
    with torch.no_grad():
        model.head_reg[-1].bias.data.fill_(train_mean)

    optim = torch.optim.AdamW(model.parameters(), lr=CFG["LR"], weight_decay=CFG["WEIGHT_DECAY"])
    amp_scaler = torch.amp.GradScaler('cuda', enabled=CFG["MIXED_PREC"] and torch.cuda.is_available())
    sched = cosine_with_warmup(optim, CFG["WARMUP_EPOCHS"], CFG["EPOCHS"])
    ema = EMA(model, decay=CFG["EMA_DECAY"])

    best_score = 1e9
    bad = 0
    history = []

    for epoch in range(1, CFG["EPOCHS"]+1):
        t0 = time.time()
        tr = train_one_epoch(model, train_loader, optim, amp_scaler, device, ema=ema)
        va = evaluate(model, val_loader, device, le, use_ema=ema, collect=False)
        sched.step()

        score = composite_score(va["mae"], va["f1_macro"], alpha=CFG["SCORE_ALPHA"])
        history.append({"epoch": epoch, **{f"tr_{k}": v for k,v in tr.items()},
                        **{f"va_{k}": v for k,v in va.items()}, "score": float(score)})

        print(
            f"[{epoch:03d}] "
            f"tr_loss={tr['loss']:.4f} tr_mae={tr['reg_mae']:.3f} "
            f"| va_mae={va['mae']:.3f} va_rmse={va['rmse']:.3f} "
            f"| va_f1M={va['f1_macro']:.3f} va_acc={va['acc']:.3f} "
            f"| score={score:.3f} | time={time.time()-t0:.1f}s"
        )

        if score < best_score:
            best_score = score; bad = 0
            ema.apply_shadow(model)
            torch.save(model.state_dict(), save_dir/"best_convnext_tiny_multi_meta.pth")
            ema.restore(model)
            with open(save_dir/"label_map.json", "w") as f:
                json.dump({i:c for i,c in enumerate(le.classes_.tolist())}, f, ensure_ascii=False, indent=2)
            with open(save_dir/"train_history.json", "w") as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
        else:
            bad += 1
            if bad >= CFG["EARLY_STOP"]:
                print(f"Early stopping at epoch {epoch}")
                break

    # ----- Test (+ 그래프 저장용 상세 예측 수집) -----
    model.load_state_dict(torch.load(save_dir/"best_convnext_tiny_multi_meta.pth", map_location=device))
    te = evaluate(model, test_loader, device, le, use_ema=None, collect=True)
    print(f"\n[Test] mae={te['mae']:.3f} rmse={te['rmse']:.3f} | f1M={te['f1_macro']:.3f} acc={te['acc']:.3f}")

    # ----- 기록/그래프 저장 -----
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(save_dir/"history.csv", index=False)

    # (요청) 이중 MAE 그래프 + 최적 에폭 점선
    plot_mae_dual(hist_df, save_dir/"curve_mae_dual.png")

    # 보조 곡선
    plot_curve(hist_df["epoch"], hist_df["tr_loss"], "Train Total Loss", "epoch", "loss", save_dir/"curve_losses_train.png")
    plot_curve(hist_df["epoch"], hist_df["va_loss"], "Val Total Loss", "epoch", "loss", save_dir/"curve_losses.png")
    plot_curve(hist_df["epoch"], hist_df["va_f1_macro"], "Val Macro-F1", "epoch", "F1", save_dir/"curve_f1.png")
    plot_curve(hist_df["epoch"], hist_df["va_acc"], "Val Accuracy", "epoch", "Acc", save_dir/"curve_acc.png")

    # 회귀 시각화
    plot_scatter(te["y_reg_true"], te["y_reg_pred"], "Test dB: Pred vs True", "True dB", "Pred dB", save_dir/"test_pred_vs_true.png")
    plot_hist(te["y_reg_true"] - te["y_reg_pred"], "Test Residuals", "True - Pred (dB)", save_dir/"test_residual_hist.png")

    # 분류 시각화
    plot_confmat(te["y_true"], te["y_pred"], te["label_map"], save_dir/"test_cm.png", normalize=True)

    # ----- 메트릭/예측 저장 (배열 직렬화 주의) -----
    np.savez(save_dir/"test_arrays.npz",
             y_true=te["y_true"], y_pred=te["y_pred"],
             y_reg_true=te["y_reg_true"], y_reg_pred=te["y_reg_pred"])

    id2label = {int(k): v for k, v in te["label_map"].items()}
    pd.DataFrame({"true_db": te["y_reg_true"], "pred_db": te["y_reg_pred"]}).to_csv(save_dir/"test_regression_preds.csv", index=False)
    cls_df = pd.DataFrame({"true_id": te["y_true"].astype(int), "pred_id": te["y_pred"].astype(int)})
    cls_df["true_label"] = cls_df["true_id"].map(id2label)
    cls_df["pred_label"] = cls_df["pred_id"].map(id2label)
    cls_df.to_csv(save_dir/"test_classification_preds.csv", index=False)

    te_save = {"mae": te["mae"], "rmse": te["rmse"], "f1_macro": te["f1_macro"], "acc": te["acc"],
               "label_map": {int(k): v for k, v in te["label_map"].items()}}
    with open(save_dir/"test_metrics.json", "w") as f:
        json.dump(te_save, f, ensure_ascii=False, indent=2)

    print("\n[완료] 모델/기록/그래프 저장:", save_dir)

if __name__ == "__main__":
    main()
