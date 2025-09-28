

import os, json, random, re
from pathlib import Path
from hashlib import md5

import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
CFG = {
    "DATA_ROOT": "/content/drive/MyDrive/kaggle_noise",  # ★ 경로 수정
    "SR": 22050,
    "DURATION": 2.5,
    "N_MELS": 128, "FMIN": 20, "FMAX": 8000,
    "N_FFT": 2048, "HOP": 512,
    "SEED": 42,
    "PLOT_EDA": True,
}
CACHE_NAME  = "logmel_cache_np"
CSV_PREFIX  = "ast_split_with_spec"

# -------- Colab Drive (무시 가능) --------
try:
    from google.colab import drive
    drive.mount('/content/drive')
except Exception:
    pass

random.seed(CFG["SEED"]); np.random.seed(CFG["SEED"])

# ---------------- CSV 로드 ----------------
DATA_ROOT = Path(CFG["DATA_ROOT"])
for n in ["train_split.csv","val_split.csv","test_split.csv"]:
    assert (DATA_ROOT/n).exists(), f"{n} 가 필요합니다. ({DATA_ROOT/n})"

train_src = pd.read_csv(DATA_ROOT/"train_split.csv")
val_src   = pd.read_csv(DATA_ROOT/"val_split.csv")
test_src  = pd.read_csv(DATA_ROOT/"test_split.csv")

# =========================
# 파일명 → 메타 파싱 (예: 2F_book_1_3_2.wav)
# =========================
def parse_meta_from_filename(wav_name:str):
    nm = Path(wav_name).stem
    parts = nm.split("_")
    floor, source, intensity, distance, repeat = None, None, None, None, None
    if len(parts) >= 5:
        p0 = parts[0].upper()
        m = re.match(r"(\d+)\s*F$", p0)
        if m: floor = int(m.group(1))
        else:
            try: floor = int(p0)
            except: floor = None
        source = parts[1]
        try: intensity = float(parts[2])
        except: intensity = None
        try: distance  = float(parts[3])
        except: distance = None
        try: repeat    = int(parts[4])
        except: repeat = None
    return floor, source, intensity, distance, repeat

# =========================
# 경로 자동 보정
# =========================
def _try_candidates(split:str, category:str, wav_file:str, wav_path:str, data_root:Path):
    category = (category or "").strip()
    wav_file = (wav_file or "").strip()
    wav_path = (wav_path or "").lstrip("/") if isinstance(wav_path, str) else ""
    candidates = []
    if wav_path:
        candidates.append(data_root / split / "data" / wav_path)
    if category and wav_file:
        candidates.append(data_root / split / "data" / category / wav_file)
    if wav_path and "/" in wav_path and wav_path.split("/", 1)[0] in ("2F", "3F"):
        no_floor = wav_path.split("/", 1)[1]
        candidates.append(data_root / split / "data" / no_floor)
    if category and wav_file:
        for fl in ("2F","3F"):
            candidates.append(data_root / split / "data" / fl / category / wav_file)
    if wav_path and "/" in wav_path and wav_path.split("/",1)[0] not in ("2F","3F"):
        for fl in ("2F","3F"):
            candidates.append(data_root / split / "data" / fl / wav_path)
    for p in candidates:
        if p.exists(): return str(p)
    return None

def normalize_split(df: pd.DataFrame, split: str, data_root: Path) -> pd.DataFrame:
    df = df.copy()

    # db
    col_db = "db" if "db" in df.columns else ("decibel" if "decibel" in df.columns else None)
    if col_db is None: raise AssertionError("CSV에 db(또는 decibel) 컬럼이 필요합니다.")
    def _to_float_db(s):
        try: return float(str(s).replace(",", "").strip())
        except: return np.nan
    db = df[col_db].map(_to_float_db).astype(np.float32)

    # cls/source
    if "cls" in df.columns: cls = df["cls"].astype(str)
    elif "category" in df.columns: cls = df["category"].astype(str)
    else: raise AssertionError("CSV에 cls(또는 category) 컬럼이 필요합니다.")

    has_fp    = "filepath" in df.columns
    has_wpath = "wav_path" in df.columns
    has_wfile = "wav_file" in df.columns

    filepaths, floors, sources, intens, dists, reps = [], [], [], [], [], []
    fixed, missing = 0, 0

    for _, row in df.iterrows():
        category = str(row["category"] if "category" in df.columns else row["cls"]) if ("category" in df.columns or "cls" in df.columns) else ""
        wav_file = str(row["wav_file"]) if has_wfile and pd.notna(row["wav_file"]) else ""
        wav_path = str(row["wav_path"]) if has_wpath and pd.notna(row["wav_path"]) else ""

        cand = None
        if has_fp and pd.notna(row.get("filepath", None)):
            p = Path(str(row["filepath"]))
            if p.exists():
                cand = str(p)
            else:
                cand = _try_candidates(split, category, wav_file, wav_path or p.as_posix(), data_root)
                if cand: fixed += 1
        else:
            cand = _try_candidates(split, category, wav_file, wav_path, data_root)

        if cand is None:
            filepaths.append("__MISSING__"); missing += 1
            floors.append(np.nan); sources.append(None); intens.append(np.nan); dists.append(np.nan); reps.append(np.nan)
            continue

        filepaths.append(cand)
        f,s,i,d,r = parse_meta_from_filename(Path(cand).name)
        floors.append(f); sources.append(s); intens.append(i); dists.append(d); reps.append(r)

    out = pd.DataFrame({
        "filepath": filepaths,
        "db": db,
        "cls": cls,
        "floor": floors,
        "source_from_name": sources,
        "intensity": intens,
        "distance": dists,
        "repeat": reps,
    }).dropna(subset=["db"])

    if missing: print(f"[경고] {split}: 자동 보정에도 불구하고 찾지 못한 파일 {missing}개")
    if fixed:   print(f"[정보] {split}: 경로 자동 보정 성공 {fixed}개")
    out = out[(out["filepath"] != "__MISSING__")].reset_index(drop=True)

    # 파일명 소음원이 있으면 cls를 파일명 기준으로 교정
    mask_fix = out["source_from_name"].notna()
    out.loc[mask_fix, "cls"] = out.loc[mask_fix, "source_from_name"].astype(str)

    print(f"[{split}] resolved rows = {len(out)} (원본 {len(df)})")
    if len(out): print(out.head())
    return out

# 표준화 & 보정
train_src = normalize_split(train_src, "train", DATA_ROOT)
val_src   = normalize_split(val_src,   "val",   DATA_ROOT)
test_src  = normalize_split(test_src,  "test",  DATA_ROOT)
assert len(train_src) and len(val_src) and len(test_src), "빈 split이 있습니다."

# ---------------- 특징 추출 (Log-Mel) ----------------
def load_audio(fp):
    y, sr = librosa.load(fp, sr=CFG["SR"], mono=True)
    T = int(CFG["SR"]*CFG["DURATION"])
    if len(y) < T: y = np.pad(y, (0, T-len(y)))
    else:          y = y[:T]
    return y, sr

def wav_to_logmel(fp):
    y, sr = load_audio(fp)
    S = librosa.feature.melspectrogram(
        y=y, sr=sr,
        n_mels=CFG["N_MELS"], fmin=CFG["FMIN"], fmax=CFG["FMAX"],
        n_fft=CFG["N_FFT"], hop_length=CFG["HOP"], power=2.0
    )
    X = librosa.power_to_db(S, ref=np.max)  # [n_mels, time]
    mn, mx = X.min(), X.max()
    X = (X - mn) / (mx - mn + 1e-8)         # [0,1]
    return X.astype(np.float32)

def spec_key(fp):
    s = f"{fp}_LM_{CFG['SR']}_{CFG['DURATION']}_{CFG['N_FFT']}_{CFG['HOP']}_{CFG['N_MELS']}_{CFG['FMIN']}_{CFG['FMAX']}"
    return md5(s.encode()).hexdigest()

CACHE_DRIVE = DATA_ROOT/"logmel_cache_np"
CACHE_DRIVE.mkdir(exist_ok=True, parents=True)

def ensure_cached(df, split):
    from tqdm import tqdm
    csv_path = DATA_ROOT/f"{CSV_PREFIX}_{split}.csv"
    spaths = []
    for fp in tqdm(df["filepath"].tolist(), desc=f"[cache] {split}"):
        npy = CACHE_DRIVE/f"{spec_key(fp)}.npy"
        if not npy.exists(): np.save(npy, wav_to_logmel(fp))
        spaths.append(str(npy))
    out = df.copy()
    out["spec_path"] = spaths
    for c in ["floor","intensity","distance","repeat","db"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    out.to_csv(csv_path, index=False)
    return out

train_df = ensure_cached(train_src, "train")
val_df   = ensure_cached(val_src,   "val")
test_df  = ensure_cached(test_src,  "test")

# ---------------- EDA 플롯 (옵션) ----------------
if CFG.get("PLOT_EDA", True):
    eda_dir = DATA_ROOT/"eda"
    eda_dir.mkdir(exist_ok=True, parents=True)

    def _bar_count(series, title, save):
        cnt = series.value_counts().sort_index()
        plt.figure()
        plt.bar(cnt.index.astype(str), cnt.values)
        plt.title(title); plt.xlabel("category"); plt.ylabel("count")
        plt.tight_layout(); plt.savefig(eda_dir/save, dpi=150); plt.close()

    def _hist(series, title, save, bins=20):
        vals = pd.to_numeric(series, errors="coerce").dropna()
        plt.figure()
        plt.hist(vals.values, bins=bins)
        plt.title(title); plt.xlabel("value"); plt.ylabel("freq")
        plt.tight_layout(); plt.savefig(eda_dir/save, dpi=150); plt.close()

    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    _bar_count(all_df["floor"].dropna().astype(int), "Floor distribution", "count_floor.png")
    _bar_count(all_df["cls"].astype(str), "Source(class) distribution", "count_source.png")
    _hist(all_df["intensity"], "Intensity histogram", "hist_intensity.png")
    _hist(all_df["distance"],  "Distance histogram",  "hist_distance.png")

# ---------------- 요약 ----------------
classes = sorted(pd.Series(train_df["cls"].astype(str).unique()).tolist())
meta = {
    "config": CFG,
    "classes": classes,
    "cache_dir": str(CACHE_DRIVE),
    "csv_prefix": CSV_PREFIX,
    "rows": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
    "num_features": ["floor","intensity","distance","repeat"]
}
with open(DATA_ROOT/"preprocess_meta.json", "w") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)

print("\n[완료] 전처리/캐시 생성")
print(" - Cache dir :", CACHE_DRIVE)
print(" - CSVs      :", [DATA_ROOT/f'{CSV_PREFIX}_train.csv',
                        DATA_ROOT/f'{CSV_PREFIX}_val.csv',
                        DATA_ROOT/f'{CSV_PREFIX}_test.csv'])
print(" - Meta      :", DATA_ROOT/'preprocess_meta.json')
