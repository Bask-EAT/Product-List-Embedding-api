# main.py — async + aiohttp + backoff + batch write + compact status
from __future__ import annotations
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Any, List, Optional, Dict
from starlette.concurrency import run_in_threadpool
from google.cloud import firestore
from google.cloud.firestore_v1 import FieldFilter
from datetime import datetime
import os
import uvicorn
import io
import json
import asyncio
import random
from datetime import datetime
from io import BytesIO
from pathlib import Path
from contextlib import asynccontextmanager
from enum import Enum
from PIL import Image
from dotenv import load_dotenv
from google.cloud.firestore_v1.vector import Vector
import aiohttp  # pip install aiohttp

# ---- 내부 모듈(동기 함수 포함 가능)
import firestore_vector_db
from multimodal_rag_system import MultimodalRAGSystem
from jina_clip_embedding import JinaCLIPEmbedding


load_dotenv()

# =========================
# Lifespan (startup/shutdown 대체)
# =========================
aiohttp_session: Optional[aiohttp.ClientSession] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI 권장 Lifespan 핸들러: startup/shutdown 대체"""
    global aiohttp_session
    timeout = aiohttp.ClientTimeout(total=30)
    aiohttp_session = aiohttp.ClientSession(timeout=timeout)
    try:
        yield
    finally:
        if aiohttp_session and not aiohttp_session.closed:
            await aiohttp_session.close()


# === 단일 app 인스턴스만 생성 ===
app = FastAPI(title="Multimodal RAG API", version="1.1", lifespan=lifespan)


# =========================
# CORS
# =========================
def _parse_origins(env_value: Optional[str]) -> List[str]:
    if not env_value:
        return []
    parts = [p.strip() for p in env_value.replace(" ", ",").split(",")]
    return [p for p in parts if p]


ALLOWED_ORIGINS = _parse_origins(os.getenv("ADMIN_ORIGINS")) or [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# 글로벌 리소스/설정
# =========================
rag_system = MultimodalRAGSystem()
embedding_model = JinaCLIPEmbedding()


BASE_DIR = Path(__file__).resolve().parent
STATUS_FILE = BASE_DIR / "index_status.json"
LOG_FILE = BASE_DIR / "index_log.txt"
WEBHOOK_FILE = BASE_DIR / "webhook_url.txt"

INDEX_TASK: Optional[asyncio.Task] = None
INDEX_CANCEL_EVENT = asyncio.Event()
INDEX_RUNNING = asyncio.Event()

BATCH_SIZE = int(os.getenv("INDEX_BATCH_SIZE", "50"))
RETRY_ATTEMPTS = int(os.getenv("RETRY_ATTEMPTS", "5"))
RETRY_BASE = float(os.getenv("RETRY_BASE_DELAY", "0.2"))
RETRY_MAX_DELAY = float(os.getenv("RETRY_MAX_DELAY", "3.0"))
STATUS_KEEP_LAST = int(os.getenv("STATUS_KEEP_LAST", "200"))


# =========================
# 공통 유틸 (비동기 파일 I/O는 threadpool)
# =========================
async def append_log_async(message: str):
    def _write():
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().isoformat()}] {message}\n")

    await run_in_threadpool(_write)


async def update_status_async(
    status: str, progress: int = 0, total: int = 0, items: Optional[List[dict]] = None
):
    status_data = {"status": status, "progress": progress, "total": total}
    if items is not None:
        status_data["items"] = items[-STATUS_KEEP_LAST:] if STATUS_KEEP_LAST > 0 else []

    def _write():
        STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with STATUS_FILE.open("w", encoding="utf-8") as f:
            json.dump(status_data, f, ensure_ascii=False, indent=2)

    await run_in_threadpool(_write)


async def send_webhook_async():
    if not aiohttp_session or not WEBHOOK_FILE.exists():
        return

    def _read():
        with WEBHOOK_FILE.open("r", encoding="utf-8") as f:
            return f.read().strip()

    url = await run_in_threadpool(_read)
    if not url:
        return
    try:
        async with aiohttp_session.post(
            url, json={"event": "indexing_completed"}
        ) as resp:
            await resp.text()
    except Exception as e:
        await append_log_async(f"Webhook failed: {e}")


# =========================
# 백오프/재시도 헬퍼
# =========================
async def async_retry(
    coro_fn,
    *args,
    attempts=RETRY_ATTEMPTS,
    base=RETRY_BASE,
    max_delay=RETRY_MAX_DELAY,
    jitter=0.1,
    exceptions=(Exception,),
    **kwargs,
):
    last = None
    for i in range(1, attempts + 1):
        try:
            return await coro_fn(*args, **kwargs)
        except exceptions as e:
            last = e
            if i == attempts:
                raise
            delay = min(max_delay, base * (2 ** (i - 1))) + random.uniform(0, jitter)
            await asyncio.sleep(delay)
    raise last


async def retry_threadpool(
    fn,
    *args,
    attempts=RETRY_ATTEMPTS,
    base=RETRY_BASE,
    max_delay=RETRY_MAX_DELAY,
    jitter=0.1,
    exceptions=(Exception,),
    **kwargs,
):
    async def _call():
        return await run_in_threadpool(fn, *args, **kwargs)

    return await async_retry(
        _call,
        attempts=attempts,
        base=base,
        max_delay=max_delay,
        jitter=jitter,
        exceptions=exceptions,
    )


# =========================
# 모델 스키마
# =========================
class SearchRequest(BaseModel):
    query: str
    top_k: int = 10


class CrossModalIn(BaseModel):
    query: str
    top_k: int = 30
    text_weight: float = 0.6
    image_weight: float = 0.4


class VectorResponse(BaseModel):
    vector: List[float]


class PriceHistoryItem(BaseModel):
    last_updated: Optional[str] = None
    original_price: Optional[str] = None
    selling_price: Optional[str] = None


class ProductResult(BaseModel):
    id: str
    product_name: Optional[str] = None
    category: Optional[str] = None
    image_url: Optional[str] = None
    product_address: Optional[str] = None
    quantity: Optional[str] = None
    out_of_stock: Optional[str] = None
    last_updated: Optional[str] = None
    is_emb: Optional[str] = None
    similarity_score: Optional[float] = None
    price: Optional[float] = None
    last_price_updated: Optional[str] = None
    price_history: Optional[List[PriceHistoryItem]] = None


class SearchResponse(BaseModel):
    results: List[ProductResult]


class CategoryCountsResponse(BaseModel):
    counts: Dict[str, int]
    total: int
    ratios: Dict[str, float]


# =========================
# price_history 뷰 제어
# =========================
class HistoryView(str, Enum):
    all = "all"  # 전체 이력(관리자용)
    latest = "latest"  # 최신 1건만(기본)
    none = "none"  # 필드 제거(최소 페이로드)


def _latest_history(hist):
    if isinstance(hist, list) and hist:
        return [hist[0]]
    return None


def apply_history_view(items, view: HistoryView):
    for it in items:
        hist = _get(it, "price_history")
        if view == HistoryView.all:
            continue
        if view == HistoryView.latest:
            _set(it, "price_history", _latest_history(hist))
        elif view == HistoryView.none:
            if isinstance(it, dict):
                it.pop("price_history", None)
            else:
                try:
                    delattr(it, "price_history")
                except Exception:
                    pass
    return items


# =========================
# 이미지 업로드 파싱
# =========================
async def read_imagefile(file: UploadFile) -> Image.Image:
    image_bytes = await file.read()

    def _open():
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")

    return await run_in_threadpool(_open)


def safe_remove_embedding(results: List[dict]):
    for item in results:
        item.pop("embedding", None)


# =========================
# 내부 취소 헬퍼
# =========================
class Cancelled(Exception):
    pass


def _check_cancel():
    if INDEX_CANCEL_EVENT.is_set():
        raise Cancelled()


# =========================
# 비동기 인덱싱 (백오프 + 배치쓰기)
# =========================
async def http_get_bytes(url: str) -> bytes:
    assert aiohttp_session is not None

    async def _get():
        async with aiohttp_session.get(url) as resp:
            if resp.status != 200:
                raise RuntimeError(f"HTTP {resp.status} for {url}")
            return await resp.read()

    return await async_retry(_get)


async def background_indexing_async():
    current_items: List[dict] = []
    total = 0
    processed = 0

    await update_status_async("running", 0, 0)
    await append_log_async("Started Firestore indexing")

    try:
        _check_cancel()
        vector_db = rag_system.vector_db
        _embedding_model = rag_system.embedding_model

        def _fetch_products():
            product_collection = vector_db.client.collection(
                vector_db.product_collection_name
            )
            # ✅ 신식 필터 API
            query = product_collection.where(filter=FieldFilter("is_emb", "==", "R"))
            docs = query.get()
            return [d.to_dict() for d in docs]

        products = await retry_threadpool(_fetch_products)
        total = len(products)

        pending: List[dict] = []

        async def commit_pending():
            nonlocal pending, processed, current_items
            if not pending:
                return

            def _commit():
                batch = vector_db.client.batch()

                # ✅ 단일/분리형 자동 분기
                split_mode = bool(
                    not vector_db.vector_collection_name
                    and vector_db.vector_collection_text
                    and vector_db.vector_collection_image
                )

                product_collection = vector_db.client.collection(
                    vector_db.product_collection_name
                )

                if split_mode:
                    text_col = vector_db.client.collection(
                        vector_db.vector_collection_text
                    )
                    img_col = vector_db.client.collection(
                        vector_db.vector_collection_image
                    )
                    for it in pending:
                        vref_t = text_col.document(it["id"])
                        vref_i = img_col.document(it["id"])
                        pref = product_collection.document(it["id"])

                        batch.set(
                            vref_t,
                            {"id": it["id"], "text_embedding": Vector(it["text_emb"])},
                        )
                        batch.set(
                            vref_i,
                            {
                                "id": it["id"],
                                "image_embedding": Vector(it["image_emb"]),
                            },
                        )
                        batch.update(pref, {"is_emb": "D"})
                else:
                    vector_collection = vector_db.client.collection(
                        vector_db.vector_collection_name
                    )
                    for it in pending:
                        vref = vector_collection.document(it["id"])
                        pref = product_collection.document(it["id"])

                        batch.set(
                            vref,
                            {
                                "id": it["id"],
                                "text_embedding": Vector(it["text_emb"]),
                                "image_embedding": Vector(it["image_emb"]),
                            },
                        )
                        batch.update(pref, {"is_emb": "D"})

                batch.commit()

            await retry_threadpool(_commit)
            pending = []

        for idx, product in enumerate(products):
            _check_cancel()

            item_status = {
                "id": product.get("id"),
                "product_name": product.get("product_name"),
                "category": product.get("category"),
                "image_url": product.get("image_url"),
                "product_address": product.get("product_address"),
                "quantity": product.get("quantity"),
                "out_of_stock": product.get("out_of_stock"),
                "last_updated": datetime.utcnow().isoformat(),
                "is_emb": product.get("is_emb"),
                "similarity_score": None,
            }

            try:
                if not product.get("product_name") or not product.get("image_url"):
                    raise ValueError("Missing required fields")

                _check_cancel()
                # ✅ 텍스트 임베딩은 문서 인덱싱 태스크로 고정
                text_emb = await retry_threadpool(
                    _embedding_model.encode_text,
                    product["product_name"],
                    "retrieval.query",
                )

                _check_cancel()

                img_bytes = await http_get_bytes(product["image_url"])
                _check_cancel()

                def _to_image():
                    return Image.open(BytesIO(img_bytes)).convert("RGB")

                image = await run_in_threadpool(_to_image)
                _check_cancel()

                image_emb = await retry_threadpool(_embedding_model.encode_image, image)
                _check_cancel()

                pending.append(
                    {"id": product["id"], "text_emb": text_emb, "image_emb": image_emb}
                )
                item_status["status"] = "✅ success"

                if len(pending) >= BATCH_SIZE:
                    _check_cancel()
                    await commit_pending()
                    _check_cancel()

            except Cancelled:
                try:
                    await commit_pending()
                except Exception as e:
                    await append_log_async(f"Pending commit on cancel failed: {e}")
                await update_status_async("stopped", idx, total, current_items)
                await append_log_async("🛑 Indexing stopped by request")
                return
            except Exception as e:
                item_status["status"] = f"❌ failed: {e}"

            current_items.append(item_status)
            processed = idx + 1
            await update_status_async("running", processed, total, current_items)
            await asyncio.sleep(0)

        await commit_pending()
        await update_status_async("completed", total, total, current_items)
        await append_log_async("✅ Indexing completed")
        await send_webhook_async()

    except Cancelled:
        await update_status_async("stopped", processed, total, current_items)
        await append_log_async("🛑 Indexing stopped by request (outer)")
    except Exception as e:
        await update_status_async("failed", processed, total, current_items)
        await append_log_async(f"❌ Indexing failed: {e}")
    finally:
        INDEX_RUNNING.clear()
        INDEX_CANCEL_EVENT.clear()


PRODUCT_COL = os.getenv("FIRESTORE_PRODUCT_COLLECTION", "emart_product")
PRICE_COL = os.getenv("FIRESTORE_PRICE_COLLECTION", "emart_price")
IN_CHUNK = 10  # ✅ Firestore 'in' 쿼리 안전 청크


# ===== 유틸: dict/객체 호환 get/set =====
def _get(item: object, key: str):
    if isinstance(item, dict):
        return item.get(key)
    return getattr(item, key, None)


def _set(item: object, key: str, value):
    if isinstance(item, dict):
        item[key] = value
    else:
        try:
            setattr(item, key, value)
        except Exception:
            pass


# ===== 파서 유틸 =====
def _parse_price_to_float(v) -> Optional[float]:
    """
    "12,340", "12340원", 12340 -> 12340.0
    소수점도 허용("123.45")
    """
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = "".join(ch for ch in str(v) if ch.isdigit() or ch == ".")
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def _parse_iso_or_none(s) -> Optional[datetime]:
    """
    다양한 ISO 포맷 허용: 2025-08-12T16:37:27.222082 / ...Z 등
    """
    if not s:
        return None
    try:
        return datetime.fromisoformat(str(s).rstrip("Z"))
    except Exception:
        return None


# ===== Firestore 조회 헬퍼 =====
def _fetch_by_doc_id(db: firestore.Client, ids: List[str]) -> dict[str, dict]:
    doc_refs = [db.collection(PRICE_COL).document(str(pid)) for pid in ids]
    snaps = db.get_all(doc_refs)
    return {s.id: (s.to_dict() or {}) for s in snaps if s.exists}


def _fetch_by_field_in(db: firestore.Client, want_ids: List[str]) -> dict[str, dict]:
    out = {}
    col = db.collection(PRICE_COL)
    for i in range(0, len(want_ids), IN_CHUNK):
        chunk = [str(x) for x in want_ids[i : i + IN_CHUNK]]
        if not chunk:
            continue
        # ✅ 신식 필터 API로 교체
        for s in col.where(filter=FieldFilter("id", "in", chunk)).stream():
            d = s.to_dict() or {}
            pid = str(d.get("id") or s.id)
            out[pid] = d
    return out


# ===== 메인: 결과 병합 =====
async def enrich_with_prices(results: List[object]) -> int:
    """
    emart_price 컬렉션을 조회해 각 결과에 아래 필드를 병합합니다.
      - price (float)
      - last_price_updated (ISO 문자열)
      - price_history (정규화·최신순 정렬)
      - quantity (상위 필드로 주입)
      - out_of_stock (상위 필드로 주입)
    dict 뿐 아니라 Pydantic 모델 객체도 지원합니다. 병합된 'price' 갯수 반환.
    """
    if not results:
        return 0

    ids = list({_get(r, "id") for r in results if _get(r, "id")})
    if not ids:
        return 0

    db = firestore.Client()
    price_map: dict[str, dict] = {}

    # 1) 문서ID == product id 우선 조회
    docid_data = await run_in_threadpool(_fetch_by_doc_id, db, ids)
    price_map.update(docid_data)

    # 2) 못찾은 것은 field 'id' IN 조회
    missing_ids = [pid for pid in ids if pid not in price_map]
    if missing_ids:
        field_data = await run_in_threadpool(_fetch_by_field_in, db, missing_ids)
        price_map.update(field_data)

    merged = 0

    for item in results:
        pid = _get(item, "id")
        if not pid:
            continue

        pdata = price_map.get(str(pid))
        if not pdata:
            continue

        # ----- price_history 정규화 -----
        history = pdata.get("price_history")
        norm_hist: List[dict[str, Any]] = []
        if isinstance(history, list) and history:
            for h in history:
                h = h or {}
                last_upd = h.get("last_updated") or h.get("updated_at") or h.get("ts")
                norm_hist.append(
                    {
                        "last_updated": str(last_upd) if last_upd is not None else None,
                        "original_price": (
                            str(h.get("original_price"))
                            if h.get("original_price") is not None
                            else None
                        ),
                        "selling_price": (
                            str(h.get("selling_price"))
                            if h.get("selling_price") is not None
                            else None
                        ),
                    }
                )

            # 최신순 정렬
            norm_hist.sort(
                key=lambda x: (_parse_iso_or_none(x["last_updated"]) or datetime.min),
                reverse=True,
            )

            # 상위에 세팅
            _set(item, "price_history", norm_hist)

        # ----- 최신값 선택 로직 -----
        latest_hist = norm_hist[0] if norm_hist else None
        hist_ts = (
            _parse_iso_or_none(latest_hist.get("last_updated")) if latest_hist else None
        )
        hist_price = None
        if latest_hist:
            hist_price = _parse_price_to_float(
                latest_hist.get("selling_price")
            ) or _parse_price_to_float(latest_hist.get("original_price"))

        flat_raw_price = pdata.get("price", pdata.get("current_price"))
        flat_price = _parse_price_to_float(flat_raw_price)
        flat_ts = _parse_iso_or_none(pdata.get("last_price_updated"))

        def _choose(a_ts, a_price, b_ts, b_price):
            if a_ts and b_ts:
                return (
                    ("hist", a_ts, a_price) if a_ts >= b_ts else ("flat", b_ts, b_price)
                )
            if a_ts:
                return ("hist", a_ts, a_price)
            if b_ts:
                return ("flat", b_ts, b_price)
            if a_price is not None:
                return ("hist", None, a_price)
            if b_price is not None:
                return ("flat", None, b_price)
            return (None, None, None)

        src, chosen_ts, chosen_price = _choose(hist_ts, hist_price, flat_ts, flat_price)

        if chosen_price is not None:
            _set(item, "price", chosen_price)
            merged += 1

        if chosen_ts:
            _set(item, "last_price_updated", chosen_ts.isoformat())
        else:
            if latest_hist and latest_hist.get("last_updated"):
                _set(item, "last_price_updated", str(latest_hist["last_updated"]))
            elif pdata.get("last_price_updated"):
                _set(item, "last_price_updated", str(pdata.get("last_price_updated")))

        # 상위 필드: quantity / out_of_stock
        if pdata.get("quantity") is not None:
            _set(item, "quantity", str(pdata.get("quantity")))
        if pdata.get("out_of_stock") is not None:
            _set(item, "out_of_stock", str(pdata.get("out_of_stock")))

    return merged


def _setup_korean_font():
    """
    Matplotlib에서 한글 깨짐 방지: 가능한 시스템 폰트를 등록하고 기본 폰트로 지정.
    Windows: Malgun Gothic, macOS: AppleGothic, Linux: Noto/Nanum 우선.
    - rcParams에 기본 폰트를 설정하고, FontProperties를 반환하여 텍스트 객체에 직접 지정 가능.
    """
    import matplotlib as mpl
    from matplotlib import font_manager, rcParams

    # 새로 설치한 폰트가 있더라도 즉시 인식되도록 캐시 리로드
    try:
        font_manager._load_fontmanager(try_read_cache=False)
    except Exception:
        pass

    # 1) 설치 이름으로 우선 탐색
    installed_names = {f.name for f in font_manager.fontManager.ttflist}
    for name in [
        "Noto Sans CJK KR",
        "Noto Sans KR",
        "Malgun Gothic",
        "Apple SD Gothic Neo",
        "AppleGothic",
        "NanumGothic",
    ]:
        if name in installed_names:
            rcParams["font.family"] = name
            rcParams["axes.unicode_minus"] = False
            return font_manager.FontProperties(family=name)

    # 2) 경로로 직접 등록
    candidate_paths = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansKR-Regular.otf",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        r"C:\Windows\Fonts\malgun.ttf",
        "/System/Library/Fonts/AppleGothic.ttf",
    ]
    for p in candidate_paths:
        try:
            if os.path.exists(p):
                font_manager.fontManager.addfont(p)
                # 등록된 폰트의 내부 이름을 알아내기 위해 Properties 생성
                fp = font_manager.FontProperties(fname=p)
                name = fp.get_name() or "sans-serif"
                rcParams["font.family"] = name
                rcParams["axes.unicode_minus"] = False
                return fp
        except Exception:
            continue

    # 3) 마지막 안전장치
    rcParams["axes.unicode_minus"] = False
    return font_manager.FontProperties()  # 기본


# =========================
# 검색 API
# =========================
@app.post("/search/text", response_model=SearchResponse)
async def search_products(
    payload: SearchRequest, history: HistoryView = Query(HistoryView.latest)
):
    try:
        results = await run_in_threadpool(
            rag_system.search_by_text, payload.query, payload.top_k
        )
        safe_remove_embedding(results)
        merged = await enrich_with_prices(results)
        apply_history_view(results, history)
        print(f"[search/text] merged={merged}, history={history}")
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/string2vec", response_model=VectorResponse)
async def string2vec(payload: SearchRequest):
    try:
        vector = await retry_threadpool(embedding_model.encode_text, payload.query)
        return {"vector": vector}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/image", response_model=SearchResponse)
async def search_by_image(
    file: UploadFile = File(...),
    top_k: int = 10,
    history: HistoryView = Query(HistoryView.latest),
):
    try:
        image = await read_imagefile(file)
        results = await run_in_threadpool(rag_system.search_by_image, image, top_k)
        safe_remove_embedding(results)
        merged = await enrich_with_prices(results)
        apply_history_view(results, history)
        print(f"[search/image] merged={merged}, history={history}")
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image search failed: {str(e)}")


@app.post("/image2vec", response_model=VectorResponse)
async def image2vec(file: UploadFile = File(...)):
    try:
        image = await read_imagefile(file)
        embedding = await retry_threadpool(embedding_model.encode_image, image)
        if embedding is None:
            raise HTTPException(status_code=500, detail="Image embedding failed.")
        return {"vector": embedding}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image vectorization failed: {e}")


@app.post("/search/multimodal", response_model=SearchResponse)
async def search_multimodal(
    query: str = Form(...),
    file: UploadFile = File(...),
    top_k: int = Form(30),
    alpha: float = Form(0.7),
    history: HistoryView = Query(HistoryView.latest),
):
    if not (0.0 <= alpha <= 1.0):
        raise HTTPException(status_code=422, detail="alpha must be between 0.0 and 1.0")
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=415, detail="Only image files are accepted")
    try:
        image = await read_imagefile(file)
        results = await run_in_threadpool(
            rag_system.search_multimodal, query, image, top_k, alpha
        )
        safe_remove_embedding(results)
        merged = await enrich_with_prices(results)
        apply_history_view(results, history)
        print(f"[search/multimodal] merged={merged}, history={history}")
        return {"results": results}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Multimodal search failed: {e}")


# ✅ 멀티모달 Late Fusion 텍스트 검색 전용
@app.post("/search/crossmodal-text", response_model=SearchResponse)
async def crossmodal_text_search(
    payload: CrossModalIn, history: HistoryView = Query(HistoryView.latest)
):
    try:
        results = await run_in_threadpool(
            rag_system.search_by_text_crossmodal,
            payload.query,
            payload.top_k,
            payload.text_weight,
            payload.image_weight,
        )
        safe_remove_embedding(results)
        merged = await enrich_with_prices(results)
        apply_history_view(results, history)
        print(f"[search/crossmodal-text] merged={merged}, history={history}")
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =========================
# 카테고리 집계 API (JSON)
# =========================
@app.get("/metrics/category-counts", response_model=CategoryCountsResponse)
async def get_category_counts(
    only_embedded: Optional[str] = Query(
        None,
        pattern="^(D|R)$",
        description="D: 임베딩 완료, R: 임베딩 대기, 미지정: 전체",
    ),
    category_field: str = Query("category", description="카테고리 필드명"),
):
    """
    Firestore에서 카테고리별 상품 갯수를 집계하여 반환합니다.
    - counts: {카테고리: 개수}
    - total: 전체 개수
    - ratios: {카테고리: 비율(0~1)}
    """
    flag = None
    if only_embedded == "D":
        flag = True
    elif only_embedded == "R":
        flag = False

    def _fetch():
        return rag_system.vector_db.get_category_distribution(
            only_embedded=flag, category_field=category_field
        )

    data = await run_in_threadpool(_fetch)
    return CategoryCountsResponse(**data)


# =========================
# 카테고리 원그래프(PNG)
# =========================
@app.get("/metrics/category-pie.png")
async def get_category_pie_png(
    only_embedded: Optional[str] = Query(
        None,
        pattern="^(D|R)$",
        description="D: 임베딩 완료, R: 임베딩 대기, 미지정: 전체",
    ),
    category_field: str = Query("category", description="카테고리 필드명"),
    show_counts: bool = Query(True, description="라벨에 개수 표시 여부"),
):
    """
    카테고리별 상품 갯수를 원그래프로 렌더링하여 PNG로 반환합니다.
    (Matplotlib 필요. 미설치 시 500 반환)
    """
    flag = None
    if only_embedded == "D":
        flag = True
    elif only_embedded == "R":
        flag = False

    def _fetch():
        return rag_system.vector_db.get_category_distribution(
            only_embedded=flag, category_field=category_field
        )

    data = await run_in_threadpool(_fetch)
    counts = data.get("counts", {})
    total = data.get("total", 0)

    if not counts or total == 0:
        raise HTTPException(status_code=404, detail="집계할 데이터가 없습니다.")

    # 안전한 런타임 임포트 + 백엔드 렌더러 준비
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)  # 헤드리스 렌더러
        # ✅ 폰트 설정(전역 rcParams + FontProperties 반환)
        font_prop = _setup_korean_font()
        import matplotlib.pyplot as plt
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Matplotlib 초기화 실패: {e}")

    labels = list(counts.keys())
    sizes = list(counts.values())

    def _autopct(p):
        if show_counts:
            return f"{p:.1f}%\n({int(round(p*total/100))})"
        return f"{p:.1f}%"

    buf = BytesIO()
    try:
        fig, ax = plt.subplots(figsize=(6.5, 6.5))
        # ✅ 라벨/퍼센트 텍스트 모두 한글 폰트 강제
        ax.pie(
            sizes,
            labels=labels,
            autopct=_autopct,
            startangle=90,
            counterclock=False,
            textprops={"fontproperties": font_prop},
        )
        ax.axis("equal")
        # ✅ 타이틀에도 한글 폰트 강제
        ax.set_title(f"카테고리별 비중 (총 {total}개)", fontproperties=font_prop)
        fig.tight_layout()
        fig.savefig(buf, format="png", dpi=160)
    finally:
        plt.close("all")

    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


# =========================
# 인덱싱 API
# =========================
@app.post("/index/start")
async def start_indexing():
    global INDEX_TASK
    if INDEX_TASK and not INDEX_TASK.done():
        return {"message": "⚠️ 이미 인덱싱이 실행 중입니다."}
    INDEX_CANCEL_EVENT.clear()
    INDEX_RUNNING.set()
    INDEX_TASK = asyncio.create_task(background_indexing_async(), name="indexing-task")
    return {"message": "✅ Indexing started (async task)"}


@app.post("/index/stop")
async def stop_indexing():
    global INDEX_TASK
    if not INDEX_TASK or INDEX_TASK.done():
        return {"message": "ℹ️ 현재 진행 중인 인덱싱이 없습니다."}
    INDEX_CANCEL_EVENT.set()
    try:
        await asyncio.wait_for(INDEX_TASK, timeout=3)
    except asyncio.TimeoutError:
        INDEX_TASK.cancel()
        try:
            await INDEX_TASK
        except asyncio.CancelledError:
            pass
    finally:
        INDEX_TASK = None
    return {"message": "🛑 인덱싱 중단됨"}


@app.get("/index/status")
async def get_indexing_status():
    try:

        def _read():
            if STATUS_FILE.exists():
                with STATUS_FILE.open("r", encoding="utf-8") as f:
                    return json.load(f)
            return {"status": "idle", "progress": 0, "total": 0, "items": []}

        data = await run_in_threadpool(_read)
        data["running"] = INDEX_RUNNING.is_set()
        data["cancel_requested"] = INDEX_CANCEL_EVENT.is_set()
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/index/logs")
async def get_indexing_logs():
    def _read():
        if not LOG_FILE.exists():
            return []
        with LOG_FILE.open("r", encoding="utf-8", errors="ignore") as f:
            return [line.rstrip("\n") for line in f]

    logs = await run_in_threadpool(_read)
    return {"logs": logs}


@app.delete("/index/logs")
async def clear_indexing_logs():
    def _clear():
        parent = LOG_FILE.parent
        if str(parent) not in ("", "."):
            parent.mkdir(parents=True, exist_ok=True)
        with LOG_FILE.open("w", encoding="utf-8"):
            pass
        return True

    await run_in_threadpool(_clear)
    return {"ok": True}


@app.post("/index/webhook")
async def register_webhook(url: str = Form(...)):
    def _write():
        WEBHOOK_FILE.parent.mkdir(parents=True, exist_ok=True)
        with WEBHOOK_FILE.open("w", encoding="utf-8") as f:
            f.write(url)

    await run_in_threadpool(_write)
    return {"message": f"✅ Webhook registered: {url}"}


# =========================
# 헬스 체크
# =========================
@app.get("/health")
async def health_check():
    return {"status": "ok"}


if __name__ == "__server__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
