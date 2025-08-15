\# 🔍 Multimodal RAG — Product List Embedding (Windows GPU Edition)

Jina CLIP v2 임베딩 + **Google Firestore Vector Search**로 **텍스트 / 이미지 / 멀티모달** 검색을 제공합니다.
Streamlit 기반 **관리 UI**와 FastAPI 기반 **백엔드 API**(비동기 인덱싱·웹훅·상태/로그·카테고리 메트릭) 포함.

> ⚠️ **보안 주의**: `keys/` 내 서비스 계정 JSON은 **절대 퍼블릭 레포지토리에 푸시 금지**. 배포 전 삭제/암호화하고 `.gitignore`로 제외하세요.

---

## 🧱 구성 요약

- **임베딩 모델**: `jinaai/jina-clip-v2` (Transformers)
- **벡터 DB**: Google Cloud Firestore (Vector field)
- **유사도**: Dot Product (내적)
- **프론트엔드(관리)**: Streamlit
- **백엔드**: FastAPI + Uvicorn (검색/인덱싱/웹훅/상태/로그/메트릭)
- **인덱싱 파이프라인**: `is_emb == 'R'` → 벡터 컬렉션 upsert → 상품 `is_emb='D'`
- **상태/로그 파일**: `index_status.json`, `index_log.txt` (+ 완료 웹훅)
- **메트릭**: 카테고리별 집계 JSON & 파이차트 PNG 엔드포인트 제공

---

## 📁 디렉터리 구조(요약)

```
.env
environment.yml                # ✅ 윈도우 GPU용 환경 정의(본문 옵션 참고)
firestore_vector_db.py
jina_clip_embedding.py
main.py                         # Streamlit 관리 UI
server.py                       # FastAPI 서버
multimodal_rag_system.py
static/style.css
keys/                           # (서비스 계정 키, 커밋 금지)
index_status.json
index_log.txt
webhook_url.txt
.github/ISSUE_TEMPLATE/
```

---

## ⚙️ 설치

### 1) 사전 준비

- Windows 10/11 (win-64)
- Python **3.10**
- NVIDIA GPU + 최신 드라이버(Studio/Game Ready)
- GCP 프로젝트 + Firestore(**Native mode**)
- 서비스 계정 키(JSON)

### 2) 환경 변수(.env)

루트에 `.env` 생성:

```dotenv
# Firebase Authentication
GOOGLE_APPLICATION_CREDENTIALS=path/to/your/service-account-key.json
GOOGLE_CLOUD_PROJECT=your-project-id

# Model Configuration
MODEL_NAME=jinaai/jina-clip-v2

# Firestore Configuration
FIRESTORE_PRODUCT_COLLECTION=product-collection
FIRESTORE_PRICE_COLLECTION=price-collection
FIRESTORE_VECTOR_COLLECTION=vector-collection

# PyTorch Configuration
TORCH_DTYPE=bfloat16

# Application Configuration
PAGE_TITLE=Multimodal RAG Search
PAGE_ICON=🔍
LAYOUT=wide

# Search Configuration
MAX_SEARCH_LIMIT=50
BATCH_SIZE=32
MAX_TEXT_LENGTH=512
IMAGE_SIZE=512,512

# Environment
ENVIRONMENT=development

```

---

## 🧩 윈도우 GPU 환경 구성 (두 가지 옵션)

### ✅ 옵션 A) **Torch 2.5.x 유지** + `torchaudio`는 **CPU 빌드**(간단/안정)

- `pytorch/torchvision`은 **CUDA 빌드(Conda)**
- `torchaudio`는 **CPU 빌드(Conda/pytorch 채널)**

`environment.yml` 예시:

```yaml
name: embedf
channels:
  - conda-forge
  - pytorch
  - nvidia
dependencies:
  - python=3.10
  - pip
  # ---- API / Serving ----
  - fastapi
  - starlette
  - uvicorn
  - aiohttp
  - python-multipart
  - pydantic=2.*
  # ---- Google Cloud / Firestore ----
  - google-cloud-firestore>=2.21
  - google-api-core
  - google-auth
  - grpcio
  # ---- Data / Math / Utils ----
  - numpy>=1.26
  - scipy>=1.11
  - pandas
  - pillow
  - matplotlib
  - python-dotenv
  - requests
  - tqdm
  - regex
  - coloredlogs
  - tenacity
  # ---- ML / Embeddings ----
  - huggingface_hub
  - safetensors
  - accelerate
  - timm>=1.0.7
  # ---- (옵션) Streamlit UI ----
  - streamlit
```

적용:

```bat
conda activate embedf

conda env create -f environment.yml

pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126

pip install transformers

pip install einops

```

## ▶️ 실행

**항상 현재 인터프리터로 실행**(PATH 혼선 방지):

```bat
python -m uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

- Swagger UI: `http://localhost:8000/docs`
- CORS 오리진: `.env`의 `ADMIN_ORIGINS`로 제어(콤마 구분)

Streamlit 관리 UI:

```bat
streamlit run main.py
```

---

## 🔌 API 개요

> 상세 파라미터는 Swagger UI(`/docs`) 참고

- `GET /health` — 헬스 체크
- `POST /search/text` — `{ "query":"키위", "top_k":10 }`
- `POST /search/image?top_k=10` — multipart `file=@image.jpg`
- `POST /search/multimodal` — multipart `query`, `file`, `alpha`, `top_k`
- `POST /string2vec` / `POST /image2vec` — 임베딩 벡터 반환
- `POST /index/start` / `POST /index/stop` — 비동기 인덱싱 제어
- `GET /index/status` — 진행률/아이템 상태
- `GET /index/logs` / `DELETE /index/logs` — 로그 조회/삭제
- `POST /index/webhook` — 완료 웹훅 등록(폼필드 `url`)
- **메트릭**

  - `GET /metrics/category-counts` — `{ counts, total, ratios }`
  - `GET /metrics/category-pie.png` — 카테고리 파이차트(PNG)

응답 `results[].price_history[]`는 최신순으로 정규화되며, `price`, `last_price_updated`, `quantity`, `out_of_stock`가 상위 필드에 병합됩니다.

---

## 🧩 Firestore 스키마

### 1) 상품 (`FIRESTORE_PRODUCT_COLLECTION`)

| 필드            | 타입        | 설명                    |
| --------------- | ----------- | ----------------------- |
| id              | string      | 상품 ID                 |
| product_name    | string      | 상품명                  |
| category        | string      | 카테고리                |
| image_url       | string      | 대표 이미지 URL         |
| product_address | string      | 상세 URL                |
| quantity        | string      | 재고/용량               |
| out_of_stock    | string      | 품절 여부               |
| last_updated    | string(ISO) | 갱신 시각               |
| is_emb          | string      | `R`(미처리) → `D`(완료) |

### 2) 가격 (`FIRESTORE_PRICE_COLLECTION`, 선택)

| 필드               | 타입          | 설명                                              |
| ------------------ | ------------- | ------------------------------------------------- |
| id                 | string        | 상품 ID(또는 문서 ID 동일)                        |
| price              | string/number | 최신 가격(평면)                                   |
| last_price_updated | string(ISO)   | 최신 가격 갱신시각                                |
| price_history      | array<object> | `{ last_updated, original_price, selling_price }` |

### 3) 벡터 (`FIRESTORE_VECTOR_COLLECTION`)

| 필드            | 타입          | 설명          |
| --------------- | ------------- | ------------- |
| id              | string        | 상품 ID       |
| text_embedding  | vector<float> | 텍스트 임베딩 |
| image_embedding | vector<float> | 이미지 임베딩 |

---

## 🧮 인덱싱 파이프라인

1. 상품 컬렉션에서 `is_emb='R'` 조회 (신식 `FieldFilter` 사용)
2. `product_name` 텍스트 임베딩, `image_url` 이미지 임베딩
3. 벡터 컬렉션 upsert(`text_embedding`, `image_embedding`) 후 상품 `is_emb='D'` 업데이트
4. 진행률/아이템 상태: `index_status.json`, 로그: `index_log.txt`
5. 완료 시 등록된 웹훅으로 `{ "event": "indexing_completed" }` POST

환경 변수: `INDEX_BATCH_SIZE`, `RETRY_ATTEMPTS`, `RETRY_BASE_DELAY`, `RETRY_MAX_DELAY`, `STATUS_KEEP_LAST`

---

## 🧠 임베딩 모델 노트

- 모델: `jinaai/jina-clip-v2`
- 디바이스 우선순위: **CUDA → CPU**(폴백)
- `TORCH_DTYPE`: `bfloat16`/`float16`/`float32` 선택
- 이미지 전처리: PIL 변환, `IMAGE_SIZE` 참조

---

## 📊 프론트엔드: 카테고리 파이차트

### A) 차트 라이브러리 사용(Chart.js 또는 ECharts)

- `GET /metrics/category-counts` 응답을 그대로 매핑
- React 예시(Chart.js/ECharts 컴포넌트는 별도 폴더 참고)

```tsx
// src/pages/CategoryMetrics.tsx (발췌)
const API_BASE = import.meta.env.VITE_API_BASE || "";
const pngUrl = useMemo(() => {
  const q = new URLSearchParams();
  if (onlyEmb !== "ALL") q.set("only_embedded", onlyEmb);
  q.set("show_counts", "true");
  return `${API_BASE}/metrics/category-pie.png?${q.toString()}`;
}, [onlyEmb]);

// 서버에서 그린 PNG 직접 임베드
<img src={pngUrl} alt="Category pie" loading="lazy" />;
```

### B) 백엔드 PNG 바로 임베드만 쓰기

```tsx
const API_BASE = import.meta.env.VITE_API_BASE || "";
<img
  src={`${API_BASE}/metrics/category-pie.png?only_embedded=D`}
  alt="Category pie"
/>;
```

---

## 🈶 Matplotlib 한글 깨짐 대처

PNG 렌더 전에 한글 폰트 지정(Windows: Malgun Gothic):

```python
from matplotlib import font_manager, rcParams
font_manager.fontManager.addfont(r"C:\\Windows\\Fonts\\malgun.ttf")
rcParams["font.family"] = "Malgun Gothic"
rcParams["axes.unicode_minus"] = False
```

Linux 컨테이너: `fonts-noto-cjk` 또는 `fonts-nanum` 설치 후 `Noto Sans CJK KR`/`NanumGothic` 지정.

---

## 🧯 트러블슈팅

**1) OMP Error #15 (libiomp5md.dll 중복)**

- 원인: OpenMP 런타임이 **두 벌** 로드됨(예: conda의 MKL + pip torch 동시)
- 해결: 본문 옵션 A/B 중 **한 계열로 통일**.
- 점검: PowerShell에서 `Get-ChildItem -Recurse -Filter libiomp5md.dll $env:CONDA_PREFIX` → **1개**만.
- 임시 회피(비권장): `KMP_DUPLICATE_LIB_OK=TRUE`.

**2) `ModuleNotFoundError: No module named 'torch'` (uvicorn spawn)**

- `python -m uvicorn server:app --reload`처럼 **현재 인터프리터**로 실행.
- `where python`과 `where uvicorn` 경로가 같은 env인지 확인.

**3) `torchaudio` CUDA 휠 버전 불일치**

- PyTorch/vision/audio는 **같은 major.minor**로 맞추세요(예: 2.7.x + 0.22.x + 2.7.x).
- 제공 가능한 cu126 버전 목록에 따라 선택(2.7.1+cu126, 2.8.0+cu126 등).

---

## 🧪 로컬 점검용 커맨드

```bat
# FastAPI
python -m uvicorn server:app --reload

# Streamlit
streamlit run main.py

# Swagger
start http://localhost:8000/docs  # Windows
```

---

## 🤝 기여

이슈는 `.github/ISSUE_TEMPLATE/` 템플릿을 활용해 등록해 주세요. PR은 **작은 단위**로, 테스트와 린트 통과를 전제로 부탁드립니다.

---

## 📝 라이선스

MIT License © 2025 Bask\:EAT
