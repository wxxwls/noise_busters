# Interfloor Noise — Colab Reproducible Pipeline (T4/Python3)


---

## 0. 실행 환경 (Colab 권장 사양)

- 런타임: **GPU (T4)**  
  - 메뉴: *런타임 → 런타임 유형 변경 → 하드웨어 가속기: GPU / GPU 유형: T4*
- Python: Colab 기본(3.x)
- CUDA/torch: Colab 기본 버전 사용(별도 설치 불필요)

> 버전 고정은 재현성에 좋지만, Colab의 torch/CUDA 호환성 이슈가 자주 발생합니다.  
> 본 README는 **Colab 기본 torch**를 사용하도록 설계했습니다.

---

## 1. 드라이브 마운트 & ZIP 업로드

둘 중 한 가지를 선택하세요.

### A안) 드라이브 사용 (권장)
1) 본인의 Google Drive 최상단에 ZIP 업로드: `MyDrive/i_kaggle_noise_dataset.zip`  
2) 아래 셀을 **그대로** 실행합니다.

```python
from google.colab import drive
drive.mount('/content/drive')

ZIP_ON_DRIVE = '/content/drive/MyDrive/i_kaggle_noise_dataset.zip'  # 경로/파일명 변경 금지
!test -f "$ZIP_ON_DRIVE" || (echo "[에러] ZIP이 드라이브에 없습니다."; exit 1)

!mkdir -p /content/data_zip && cp "$ZIP_ON_DRIVE" /content/data_zip/
!unzip -q -o /content/data_zip/i_kaggle_noise_dataset.zip -d /content/data

!ls -R /content/data | head -n 200
```

### B안) 파일 직접 업로드
1) Colab 좌측 파일탐색기 → **Upload** 버튼으로 ZIP 업로드  
2) 아래 셀을 실행합니다.

```python
ZIP_LOCAL = '/content/i_kaggle_noise_dataset.zip'  # 좌측 업로드 후 경로 확인
!test -f "$ZIP_LOCAL" || (echo "[에러] ZIP이 업로드되지 않았습니다."; exit 1)

!mkdir -p /content/data_zip && cp "$ZIP_LOCAL" /content/data_zip/
!unzip -q -o /content/data_zip/i_kaggle_noise_dataset.zip -d /content/data

!ls -R /content/data | head -n 200
```

---

## 2. 데이터 구조 확인 (엄격 검증)

ZIP 내부는 **아래 구조**를 만족해야 합니다. (다르면 **그대로**는 실행되지 않습니다)

```
data/
  train_split.csv
  val_split.csv
  test_split.csv
  train/data/...  
  val/data/...    
  test/data/...   
```

검증 셀을 실행해 폴더/파일 유무를 체크합니다.

```python
import os
from pathlib import Path

DATA_ROOT = Path('/content/data')

required = [
    DATA_ROOT/'train_split.csv',
    DATA_ROOT/'val_split.csv',
    DATA_ROOT/'test_split.csv',
    DATA_ROOT/'train'/'data',
]

missing = [str(p) for p in required if not p.exists()]
if missing:
    raise SystemExit(f"[에러] 필수 파일/폴더가 누락되었습니다:\n- " + "\n- ".join(missing))
else:
    print("[확인] 기본 파일/폴더 존재:", *[p.name for p in required], sep="\n - ")
```

---

## 3. 패키지 설치 (최소/안정 구성)

Colab 기본 torch 사용, 오디오/스펙 관련 필수 라이브러리만 설치합니다.  
**libsndfile** 누락으로 `soundfile` 오류가 발생하지 않도록 시스템 라이브러리도 설치합니다.

```bash
%%bash
set -e
apt-get update -qq
apt-get install -y -qq libsndfile1 ffmpeg
pip install -q --upgrade pip
pip install -q numpy==1.26.4 pandas==2.2.2 scipy==1.13.1 scikit-learn==1.5.1
pip install -q librosa==0.10.2.post1 soundfile==0.12.1
pip install -q matplotlib==3.8.4 tqdm==4.66.4
pip install -q timm==1.0.9
```

> 위 버전은 T4·Colab 환경에서 검증된 안정 조합입니다.

---

## 4. 재현성(Seed) 고정

다음 셀을 **가장 먼저** 실행해서 난수/연산 순서를 고정합니다.

```python
import os, random, numpy as np, torch

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# 연산 결정론적 설정
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print("[SEED] 고정 완료:", SEED)
```

---

## 5. 전처리 실행 (Log-Mel 캐시 & 메타 병합)

> 아래 스크립트/셀은 ZIP에 포함된 코드 기준입니다.  
> **파일명이 다른 경우**, *파일명만* 바꾼 뒤 나머지는 그대로 실행하세요.

### 5-1) 노트북 사용 시
- 열어서 **[전처리]** 섹션의 셀을 순서대로 실행합니다.  
- 실행 결과로 `ast_split_with_spec_{split}.csv`와 `logmel_cache_np/*.npy`가 생성됩니다.

### 5-2) 파이썬 스크립트 사용 시

```bash
python /content/preprocess_noise_data.py \
  --data_root /content/data \
  --sr 22050 --duration 2.5 \
  --n_mels 128 --fmin 20 --fmax 8000 \
  --n_fft 2048 --hop_length 512 \
  --seed 42 \
  --cache_dir /content/logmel_cache_np \
  --out_csv_dir /content/prep_csv
```

성공 검증:

```python
from pathlib import Path
chk = [
    Path('/content/prep_csv/ast_split_with_spec_train.csv'),
    Path('/content/prep_csv/ast_split_with_spec_val.csv'),
]
missing = [str(p) for p in chk if not p.exists()]
if missing:
    raise SystemExit("[에러] 전처리 산출물이 생성되지 않았습니다:\n- " + "\n- ".join(missing))
print("[확인] 전처리 산출물 OK")
```

---

## 6. 학습/추론 실행 (멀티태스크 스태킹)

ZIP에 포함된 학습 스크립트(예: `train_stack_multitask.py`) 또는 노트북의 **[훈련/추론]** 섹션을 실행합니다.  
Colab에서 **메모리 OOM**을 피하기 위해 배치와 워커 수는 아래처럼 보수적으로 시작합니다.

```bash
python /content/train_stack_multitask.py \
  --data_root /content/data \
  --prep_csv_dir /content/prep_csv \
  --cache_dir /content/logmel_cache_np \
  --model_list effb0 convnext_tiny \
  --epochs 30 \
  --batch_size 16 \
  --num_workers 2 \
  --lr 1e-3 \
  --seed 42 \
  --out_dir /content/outputs
```

성공 시 산출물(예시):
- `/content/outputs/ckpts/…`
- `/content/outputs/logs/history.csv`
- `/content/outputs/metrics.json`
- `/content/outputs/preds_{val|test}.csv`
- `/content/outputs/figs/*.png` (학습 곡선, 혼동행렬 등)

---

## 7. 결과 동일성(재현) 확인 체크리스트

다음 항목이 **모두 동일**해야 결과가 재현됩니다.

1) **ZIP 동일성**: 동일한 `train/val/test_split.csv`와 동일한 음원 파일들  
2) **Seed/결정론 설정 유지**: 본 README의 **[4. 재현성(Seed) 고정]** 셀 먼저 실행  
3) **버전 동일**: **[3. 패키지 설치]** 버전 그대로
4) **하이퍼파라미터 동일**: 학습 스크립트 인자 동일
5) **GPU 동일성**: T4 사용 (다른 GPU는 부동소수 오차 차이가 발생할 수 있음)

간단한 해시 검증(선택):

```python
import hashlib, json, pandas as pd
def md5_file(path): 
    m = hashlib.md5(); 
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""): m.update(chunk)
    return m.hexdigest()

print("train_split.csv MD5:", md5_file("/content/data/train_split.csv"))
print("val_split.csv   MD5:", md5_file("/content/data/val_split.csv"))
print("test_split.csv  MD5:", md5_file("/content/data/test_split.csv"))

# 모델 출력의 키 파일(e.g., metrics.json) 해시도 비교
print("metrics.json MD5:", md5_file("/content/outputs/metrics.json"))

```

---

## 8. 실행 순서 요약 (체크리스트)

1) **런타임 설정**: GPU(T4) 선택  
2) **ZIP 업로드**: 드라이브 또는 로컬 업로드  
3) **압축 해제**: `/content/data`에 풀기  
4) **데이터 구조 검증** 셀 실행  
5) **패키지 설치** 셀 실행  
6) **SEED 고정** 셀 실행  
7) **전처리 실행** (캐시/CSV 생성)  
8) **학습/추론 실행**  
9) **산출물/재현 검증** (MD5 등 선택)

모든 단계는 **상단에서 하단으로 한 단계씩** 실행하세요.  
중간에 오류가 나면 **해당 단계만** 다시 확인하고, 수정 후 **다음 단계로** 진행합니다.

---

## 라이선스

- 본 프로젝트 산출물은 **MIT 계열 Permissive 라이선스**로 제출 가능합니다.  
- 데이터셋 자체의 사용 조건은 원 저작권/이용약관을 따릅니다.
