
# 🔍 Multimodal RAG — Product List Embedding

Jina CLIP v2 임베딩 + **Google Firestore Vector Search**를 이용해 **텍스트 / 이미지 / 멀티모달** 검색을 제공하는 프로젝트입니다.  
Streamlit 기반 **관리 UI**와 FastAPI 기반 **백엔드 API**(비동기 인덱싱·웹훅·상태/로그 포함)를 함께 제공합니다.

> ⚠️ 보안 주의: `keys/` 내 서비스 계정 JSON은 **절대 퍼블릭 레포지토리에 푸시하지 마세요.** 배포 전 꼭 삭제/암호화하고 `.gitignore`로 제외하세요.

---

## 🧱 구성요약

- **임베딩 모델:** `jinaai/jina-clip-v2` (Hugging Face Transformers)
- **벡터 DB:** Google Cloud Firestore (Vector field, `google-cloud-firestore`)
- **유사도:** Dot Product (내적)
- **프론트엔드:** Streamlit (검색/데이터 관리 대시보드)
- **백엔드:** FastAPI + Uvicorn (검색 API, 비동기 인덱싱, 상태/로그, 웹훅)
- **배치 인덱싱:** `is_emb == 'R'` 대상만 처리 → 벡터 컬렉션 upsert → 상품 `is_emb='D'` 업데이트
- **상태/로그 파일:** `index_status.json`, `index_log.txt` (웹훅 완료 통지 지원)

---

## 📁 디렉터리 구조

```
  .env
  .env.example
  .gitignore
  LICENSE
  README.md
  firestore_vector_db.py
  index_log.txt
  index_status.json
  jina_clip_embedding.py
  main.py
  multimodal_rag_system.py
  requirements-conda.txt
  requirements-pip.txt
  requirements.txt
  server.py
  utils.py
  webhook_url.txt
  .github/
    ISSUE_TEMPLATE/
      bug_report.md
      task.md
  keys/
    (service-account key .json)
  static/
    style.css
```

> `keys/` 경로는 실제 키 파일명을 표시하지 않습니다. 서비스 계정 JSON을 두더라도 **커밋 금지**를 권장합니다.

---

## ⚙️ 설치

### 1) 사전 준비
- Python **3.10+**
- Google Cloud 프로젝트 (Firestore **Native mode**)
- 서비스 계정 키(JSON) 발급
- (선택) NVIDIA GPU + CUDA (PyTorch)

### 2) 의존성 설치
```bash
# 가상환경 권장
conda create -n <환경이름> python=3.10

# 백엔드/프론트 공통 의존성
FastAPI 서버 포함 패키지 설치
pip install -r requirements-pip.txt
conda install -c conda-forge "pillow>=10.0.0" "numpy>=1.24.0" pytest


```

### 3) 환경 변수 설정
`.env` 파일을 루트에 생성하고 아래 예시를 채웁니다:

```dotenv
# Firebase Authentication
GOOGLE_APPLICATION_CREDENTIALS=path/to/your/service-account-key.json
GOOGLE_CLOUD_PROJECT=your-project-id

# Model Configuration
MODEL_NAME=jinaai/jina-clip-v2

# Firestore Configuration
FIRESTORE_PRODUCT_COLLECTION=product-collection
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

> **중요 변수**
> - `GOOGLE_APPLICATION_CREDENTIALS`: 서비스 계정 JSON 파일 경로
> - `GOOGLE_CLOUD_PROJECT`: GCP 프로젝트 ID
> - `FIRESTORE_PRODUCT_COLLECTION`, `FIRESTORE_VECTOR_COLLECTION`, (`FIRESTORE_PRICE_COLLECTION` 선택)
> - `MODEL_NAME=jinaai/jina-clip-v2`
> - `TORCH_DTYPE=bfloat16` (GPU/환경에 맞춰 `float16`/`float32`로 조정 가능)

---

## ▶️ 실행

### A) Streamlit 관리 UI
```bash
streamlit run main.py
```
기능:
- 텍스트/이미지/멀티모달 검색
- 결과 미리보기 (상품 썸네일/메타 정보)
- Firestore 인덱싱 트리거/상태 확인(프로젝트별 구현)

### B) FastAPI 백엔드
```bash
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```
- CORS: `ADMIN_ORIGINS`(콤마 구분) 환경변수로 허용 오리진 제어
- 상태/로그/웹훅/비동기 인덱싱 엔드포인트 포함

> 참고: 별도로 생성해둔 **OpenAPI 스펙** 및 **Postman 컬렉션**도 제공합니다.  
> - OpenAPI: `openapi.yaml` ([다운로드](sandbox:/mnt/data/openapi.yaml))  
> - Postman: `postman_collection.json` ([다운로드](sandbox:/mnt/data/postman_collection.json))

---

## 🔌 API 개요 (요약)

> 전체 스펙은 위 OpenAPI/포스트맨 파일 참고.

- `GET /health` — 헬스 체크
- `POST /search/text` — 본문: `{{"query":"키위","top_k":10}}` → 상품 리스트
- `POST /search/image?top_k=10` — 멀티파트 `file=@image.jpg` → 상품 리스트
- `POST /search/multimodal` — 멀티파트 `query=text`, `file=@image.jpg`, `alpha=0.7`, `top_k=30`
- `POST /string2vec` — 텍스트 임베딩 → 벡터(float[])
- `POST /image2vec` — 이미지 임베딩 → 벡터(float[])
- `POST /index/start` / `POST /index/stop` — 비동기 인덱싱 제어
- `GET /index/status` — 인덱싱 진행률/아이템 상태
- `GET /index/logs` / `DELETE /index/logs` — 로그 조회/삭제
- `POST /index/webhook` — 완료 웹훅 등록(폼필드 `url`)

응답 `results[].price_history[]`는 최신순으로 정규화되며, `price`, `last_price_updated`, `quantity`, `out_of_stock`가 상위 필드에 병합됩니다.

---

## 🧩 Firestore 스키마 (권장)

### 1) 상품 컬렉션 (`FIRESTORE_PRODUCT_COLLECTION`)
| 필드 | 타입 | 설명 |
|---|---|---|
| id | string | 상품 고유 ID |
| product_name | string | 상품명 |
| category | string | 카테고리 |
| image_url | string | 대표 이미지 URL |
| product_address | string | 상품 상세 URL |
| quantity | string | 재고/용량 등 |
| out_of_stock | string | 품절 여부 플래그 |
| last_updated | string(ISO) | 상품 정보 갱신시각 |
| is_emb | string | `R`(미처리) → `D`(처리 완료) |

### 2) 가격 컬렉션 (`FIRESTORE_PRICE_COLLECTION`, 선택)
| 필드 | 타입 | 설명 |
|---|---|---|
| id | string | 상품 ID (또는 문서 ID 동일) |
| price | string/number | 최신 가격(평면 필드) |
| last_price_updated | string(ISO) | 최신 가격 갱신시각 |
| price_history | array<object> | 시계열 가격 `{{last_updated, original_price, selling_price}}` |

### 3) 벡터 컬렉션 (`FIRESTORE_VECTOR_COLLECTION`)
| 필드 | 타입 | 설명 |
|---|---|---|
| id | string | 상품 ID |
| text_embedding | vector<float> | 텍스트 임베딩 |
| image_embedding | vector<float> | 이미지 임베딩 |

> Firestore **Vector Search** 사용: `google.cloud.firestore_v1.vector.Vector` 필드로 저장/검색.

---

## 🧮 인덱싱 파이프라인

1. 상품 컬렉션에서 `is_emb='R'` 문서 조회
2. `product_name` 텍스트 임베딩, `image_url` 이미지 임베딩
3. 벡터 컬렉션 upsert (`text_embedding`, `image_embedding`) 후 상품 `is_emb='D'` 업데이트
4. 진행률/아이템별 상태는 `index_status.json`, 로그는 `index_log.txt`
5. 완료 시 등록된 웹훅으로 `{{"event": "indexing_completed"}}` POST

환경변수:
- `INDEX_BATCH_SIZE`(기본 50), `RETRY_ATTEMPTS`(5), `RETRY_BASE_DELAY`(0.2), `RETRY_MAX_DELAY`(3.0), `STATUS_KEEP_LAST`(200)

---

## 🧠 임베딩 모델

- 모델: `jinaai/jina-clip-v2`
- 디바이스 우선순위: CUDA → (선택)MPS → CPU (코드에 폴백 처리)
- `TORCH_DTYPE`: `bfloat16`/`float16`/`float32` 선택 (환경에 맞게)
- 이미지 전처리: PIL 변환, 사이즈는 `.env`의 `IMAGE_SIZE` 참고

> GPU가 없거나 메모리 부족 시 자동으로 CPU/낮은 정밀도로 폴백되도록 구현되어 있습니다(코드 기준).

---

## 🔐 보안 체크리스트

- [ ] `keys/` 내 JSON 키는 **절대 커밋 금지**
- [ ] `.env`에 민감정보 포함 시 **배포 전 제거**
- [ ] 배포 환경에선 **Secret Manager** 또는 안전한 비밀 주입 사용
- [ ] CORS 오리진(`ADMIN_ORIGINS`) 제한

---

## 🧪 로컬 점검용 커맨드

```bash
# FastAPI
uvicorn server:app --reload

# Streamlit
streamlit run main.py

# (선택) 포트 열람
open http://localhost:8000/docs  # macOS
start http://localhost:8000/docs # Windows
```

---

## 🧭 로드맵(제안)

- [ ] Dockerfile / docker-compose로 **로컬-스테이징-프로덕션** 표준화
- [ ] 이미지 임베딩 다운로드를 **비동기(AIOHTTP) + 캐싱** 처리
- [ ] 지연/실패율 모니터링 (OpenTelemetry/Prometheus)
- [ ] CI: Ruff/Black/Mypy + pytest 최소 시나리오
- [ ] 백엔드 OpenAPI 자동 배포 (Swagger UI 및 ReDoc)

---

## 🤝 기여

이슈는 `.github/ISSUE_TEMPLATE/` 템플릿을 활용해 등록해 주세요. PR은 **작은 단위**로, 테스트와 린트 통과를 전제로 부탁드립니다.

---

## 📝 라이선스

MIT License © 2025 Bask:EAT
