# firestore_vector_db.py
from __future__ import annotations

import os
import logging
from pathlib import Path
from io import BytesIO
from typing import List, Dict, Optional, Callable, Tuple
from collections import defaultdict

import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry  # ✅ 올바른 위치
from PIL import Image
from dotenv import load_dotenv

from google.oauth2 import service_account
from google.cloud import firestore
from google.cloud.firestore_v1 import FieldFilter  # 신식 필터
from google.cloud.firestore_v1.vector import Vector
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure

# 선택적 정규화/점수 유틸
try:
    from utils_vector import l2_normalize, dot_score
except Exception:
    l2_normalize = None

    def dot_score(a, b):
        a_np = np.array(a, dtype=np.float32)
        b_np = np.array(b, dtype=np.float32)
        return float(np.dot(a_np, b_np))


load_dotenv()


def _build_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    s.mount("http://", HTTPAdapter(max_retries=retries))
    s.mount("https://", HTTPAdapter(max_retries=retries))
    return s


class FirestoreVectorDB:
    """Firestore vector database client (backend-safe, no Streamlit hard dependency)."""

    def __init__(
        self,
        notify: Optional[Callable[[str, str], None]] = None,  # notify(level, msg)
        logger: Optional[logging.Logger] = None,
    ):
        self._notify = notify or (lambda level, msg: None)
        self.log = logger or logging.getLogger("firestore_vector_db")

        self.client: Optional[firestore.Client] = None
        self.product_collection_name = os.getenv("FIRESTORE_PRODUCT_COLLECTION")
        # 단일 컬렉션(두 필드 공존) 모드
        self.vector_collection_name = os.getenv("FIRESTORE_VECTOR_COLLECTION")
        # 분리형 컬렉션 모드(환경변수 있으면 자동 전환)
        self.vector_collection_text = os.getenv("FIRESTORE_VECTOR_COLLECTION_TEXT")
        self.vector_collection_image = os.getenv("FIRESTORE_VECTOR_COLLECTION_IMAGE")

        # 쿼리/저장 시 L2 정규화 활성화 여부 (기본: 비활성)
        self.enable_l2_norm = os.getenv("ENABLE_L2_NORMALIZE", "false").lower() in (
            "1",
            "true",
            "yes",
        )

        self._http = _build_session()
        self.initialize_client()

    # ----------------------
    # Internal notify helpers
    # ----------------------
    def _ninfo(self, msg: str):
        self.log.info(msg)
        self._notify("info", msg)

    def _nerror(self, msg: str):
        self.log.error(msg)
        self._notify("error", msg)

    # ----------------------
    # Init & connection
    # ----------------------
    def initialize_client(self):
        """Initialize Firestore client with authentication (ENV 우선, 안전한 경로 처리)"""
        try:
            creds_path_env = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            if not creds_path_env:
                creds_path_env = "/keys/sa.json"

            p = Path(os.path.expandvars(creds_path_env)).expanduser().resolve()
            if not p.is_file():
                cwd = Path.cwd()
                self._nerror(
                    "❌ Firestore credentials not found\n"
                    f"  - GOOGLE_APPLICATION_CREDENTIALS(raw): {creds_path_env}\n"
                    f"  - Resolved path: {p}\n"
                    f"  - CWD: {cwd}\n"
                    "  - Hint: 로컬 실행이면 절대경로로 GOOGLE_APPLICATION_CREDENTIALS 지정,\n"
                    "          Docker면 ./keys를 /keys로 마운트했는지 확인."
                )
                raise ValueError(f"Invalid credentials file: {p}")

            creds = service_account.Credentials.from_service_account_file(str(p))

            project_id_env = os.getenv("GOOGLE_CLOUD_PROJECT")
            project_id = project_id_env or creds.project_id
            if not project_id:
                raise ValueError(
                    "Project ID is not set. "
                    "Set GOOGLE_CLOUD_PROJECT or ensure credentials contains project_id."
                )

            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(p)

            if not self.product_collection_name:
                raise ValueError(
                    "FIRESTORE_PRODUCT_COLLECTION environment variable not set"
                )

            # 단일/분리형 어느 쪽이든 하나는 있어야 동작
            if not self.vector_collection_name and not (
                self.vector_collection_text and self.vector_collection_image
            ):
                raise ValueError(
                    "Vector collection not configured. "
                    "Set FIRESTORE_VECTOR_COLLECTION (single) or "
                    "FIRESTORE_VECTOR_COLLECTION_TEXT/IMAGE (split)."
                )

            self.client = firestore.Client(project=project_id, credentials=creds)
            mode = (
                "SPLIT"
                if (self.vector_collection_text and self.vector_collection_image)
                else "SINGLE"
            )
            self._ninfo(
                "✅ Firestore initialized\n"
                f"  - Project: {project_id}\n"
                f"  - Creds: {p}\n"
                f"  - Product Collection: {self.product_collection_name}\n"
                f"  - Vector Mode: {mode}\n"
                f"  - Single: {self.vector_collection_name}\n"
                f"  - Split.Text: {self.vector_collection_text}\n"
                f"  - Split.Image: {self.vector_collection_image}\n"
                f"  - L2 Normalize: {self.enable_l2_norm}"
            )

            self.test_connection()

        except Exception as e:
            self._nerror(f"❌ Firestore initialization failed: {e}")
            raise

    def test_connection(self):
        """Test Firestore connection"""
        try:
            assert self.client is not None
            collections = list(self.client.collections())
            self._ninfo(
                f"✅ Firestore connection verified - {len(collections)} collections found"
            )
        except Exception as e:
            self._nerror(f"❌ Firestore connection test failed: {e}")
            raise

    # ----------------------
    # Indexing
    # ----------------------
    def store_products(
        self,
        embedding_model,
        progress_cb: Optional[Callable[[float], None]] = None,  # 0.0~1.0
        text_task: Optional[str] = "retrieval.query",
        image_timeout_sec: int = 10,
    ) -> bool:
        """
        Store products with embeddings in Firestore.
        - text_task 기본값: 'retrieval.query'
        - 모델이 해당 task를 지원하지 않으면 자동으로 None으로 폴백합니다.
        """
        try:
            assert self.client is not None
            product_collection = self.client.collection(self.product_collection_name)

            query = product_collection.where(filter=FieldFilter("is_emb", "==", "R"))
            docs = query.get()
            products_data = [doc.to_dict() for doc in docs]

            total = len(products_data)
            if total == 0:
                self._ninfo("ℹ️ No products to index (is_emb == 'R')")
                if progress_cb:
                    progress_cb(1.0)
                return True

            split_mode = bool(
                not self.vector_collection_name
                and self.vector_collection_text
                and self.vector_collection_image
            )
            stored_count = 0

            for i, product in enumerate(products_data):
                try:
                    # ---- Text embedding
                    product_name = product.get("product_name") or ""
                    if not product_name:
                        raise ValueError("product_name is empty")

                    try:
                        text_emb = embedding_model.encode_text(
                            product_name, task=text_task
                        )
                    except Exception:
                        # 모델이 task를 거부하면 task=None으로 재시도
                        text_emb = embedding_model.encode_text(product_name, task=None)
                    if text_emb is None:
                        raise RuntimeError("text embedding failed")

                    # 선택적 정규화
                    if self.enable_l2_norm and l2_normalize:
                        text_emb = l2_normalize(text_emb)

                    # ---- Image embedding
                    image_url = product.get("image_url") or ""
                    if not image_url:
                        raise ValueError("image_url is empty")

                    resp = self._http.get(image_url, timeout=image_timeout_sec)
                    resp.raise_for_status()
                    image = Image.open(BytesIO(resp.content)).convert("RGB")

                    image_emb = embedding_model.encode_image(image)
                    if image_emb is None:
                        raise RuntimeError("image embedding failed")

                    if self.enable_l2_norm and l2_normalize:
                        image_emb = l2_normalize(image_emb)

                    # ---- Write embeddings
                    pid = product.get("id")
                    if not pid:
                        raise ValueError("product id is missing")

                    batch = self.client.batch()
                    if split_mode:
                        # 분리형: text / image 각각 별도 문서
                        text_col = self.client.collection(self.vector_collection_text)
                        img_col = self.client.collection(self.vector_collection_image)
                        batch.set(
                            text_col.document(pid),
                            {"id": pid, "text_embedding": Vector(text_emb)},
                        )
                        batch.set(
                            img_col.document(pid),
                            {"id": pid, "image_embedding": Vector(image_emb)},
                        )
                    else:
                        # 단일 컬렉션: 두 필드 공존
                        vec_col = self.client.collection(self.vector_collection_name)
                        batch.set(
                            vec_col.document(pid),
                            {
                                "id": pid,
                                "text_embedding": Vector(text_emb),
                                "image_embedding": Vector(image_emb),
                            },
                        )

                    batch.update(product_collection.document(pid), {"is_emb": "D"})
                    batch.commit()

                    stored_count += 1
                    self.log.debug("Stored embeddings for product id=%s", pid)

                except Exception as ex:
                    self.log.exception("Embedding/store failed for product: %s", ex)

                if progress_cb:
                    progress_cb((i + 1) / total)

            self._ninfo(f"✅ Successfully stored {stored_count}/{total} products")
            return True

        except Exception as e:
            self._nerror(f"❌ Failed to store products: {e}")
            return False

    # ----------------------
    # Aggregation
    # ----------------------
    def get_category_counts(
        self,
        only_embedded: Optional[
            bool
        ] = None,  # True: is_emb == "D", False: "R", None: 전체
        category_field: str = "category",
    ) -> Dict[str, int]:
        assert self.client is not None
        col = self.client.collection(self.product_collection_name)

        q = col
        if only_embedded is True:
            q = q.where(filter=FieldFilter("is_emb", "==", "D"))
        elif only_embedded is False:
            q = q.where(filter=FieldFilter("is_emb", "==", "R"))

        counts: Dict[str, int] = defaultdict(int)
        for doc in q.stream():
            data = doc.to_dict() or {}
            cat = data.get(category_field) or "UNKNOWN"
            counts[str(cat)] += 1

        return dict(sorted(counts.items(), key=lambda kv: kv[0].lower()))

    def get_category_distribution(
        self,
        only_embedded: Optional[bool] = None,
        category_field: str = "category",
    ) -> Dict[str, object]:
        counts = self.get_category_counts(
            only_embedded=only_embedded, category_field=category_field
        )
        total = sum(counts.values())
        ratios = {k: (v / total if total else 0.0) for k, v in counts.items()}
        return {"counts": counts, "total": total, "ratios": ratios}

    # ----------------------
    # Vector search (single-collection mode helper)
    # ----------------------
    def vector_search(
        self, query_embedding: List[float], limit: int = 30, query_type: str = "text"
    ) -> List[Dict]:
        """Perform vector similarity search using dot product"""
        try:
            assert self.client is not None
            if not self.vector_collection_name:
                # 단일 컬렉션이 없는 경우(분리형만 있는 경우) 이 메서드는 사용 안 함
                self._nerror(
                    "vector_search() called without SINGLE vector collection configured."
                )
                return []

            collection = self.client.collection(self.vector_collection_name)
            product_collection = self.client.collection(self.product_collection_name)

            # 쿼리 정규화(옵션)
            q_emb = (
                l2_normalize(query_embedding)
                if (self.enable_l2_norm and l2_normalize)
                else query_embedding
            )
            query_vector = Vector(q_emb)

            vector_field_name = (
                "image_embedding" if query_type == "image" else "text_embedding"
            )

            vector_query = collection.find_nearest(
                vector_field=vector_field_name,
                query_vector=query_vector,
                distance_measure=DistanceMeasure.DOT_PRODUCT,
                limit=limit,
            )

            results: List[Dict] = []
            raw = list(vector_query.stream())
            if not raw:
                return results

            ids = []
            for doc in raw:
                doc_data = doc.to_dict() or {}
                product_id = doc_data.get("id") or doc.id  # ✅ 폴백
                emb = doc_data.get(vector_field_name)

                # Vector -> list 변환
                if not isinstance(emb, list):
                    try:
                        emb = list(emb)
                    except Exception:
                        emb = None

                similarity_score = 0.0
                if emb:
                    similarity_score = dot_score(q_emb, emb)

                results.append(
                    {
                        "id": product_id,
                        "similarity_score": similarity_score,
                    }
                )
                if product_id:
                    ids.append(product_collection.document(product_id))

            # 배치 조인 (client.get_all)
            joined: Dict[str, Dict] = {}
            if ids:
                for pd in self.client.get_all(ids):  # ✅ 교정
                    if pd.exists:
                        joined[pd.id] = pd.to_dict()

            final = []
            for r in results:
                meta = joined.get(r["id"], {})
                final.append({**r, **meta})

            final.sort(key=lambda x: x.get("similarity_score", 0.0), reverse=True)
            return final

        except Exception as e:
            self._nerror(f"❌ Vector search failed: {e}")
            return []

    # ----------------------
    # Vector search in specific collection (split-mode helper)
    # ----------------------
    def _vector_search_in(
        self,
        collection_name: str,
        vector_field: str,
        query_embedding: List[float],
        limit: int,
    ) -> List[Dict]:
        """분리형 컬렉션에서 벡터 검색 (id, similarity_score만 반환)"""
        assert self.client is not None
        col = self.client.collection(collection_name)

        q_emb = (
            l2_normalize(query_embedding)
            if (self.enable_l2_norm and l2_normalize)
            else query_embedding
        )
        qv = Vector(q_emb)
        vq = col.find_nearest(
            vector_field=vector_field,
            query_vector=qv,
            distance_measure=DistanceMeasure.DOT_PRODUCT,
            limit=limit,
        )
        out: List[Dict] = []
        for doc in vq.stream():
            d = doc.to_dict() or {}
            pid = d.get("id") or doc.id

            emb = d.get(vector_field)
            if not isinstance(emb, list):
                try:
                    emb = list(emb)
                except Exception:
                    emb = None

            score = dot_score(q_emb, emb) if emb else 0.0
            out.append({"id": pid, "similarity_score": score})
        return out

    # ----------------------
    # Cross-modal (Late Fusion): 텍스트 쿼리 1개로 텍스트+이미지 동시 활용
    # ----------------------
    def vector_search_crossmodal_text(
        self,
        text_query_embedding: List[float],
        limit: int = 30,
        weights: Tuple[float, float] = (0.6, 0.4),  # (text_w, image_w)
    ) -> List[Dict]:
        """
        텍스트 쿼리 1개로 text/image 각각 검색 후 가중 합산하여 최종 랭킹 반환.
        - 단일 컬렉션 모드: self.vector_search()를 text/image 각각 호출
        - 분리형 컬렉션 모드: _vector_search_in()으로 각각 호출
        """
        try:
            assert self.client is not None
            wt, wi = weights
            k = max(limit * 2, 50)

            results_text: List[Dict] = []
            results_image: List[Dict] = []

            if self.vector_collection_text and self.vector_collection_image:
                # 분리형
                results_text = self._vector_search_in(
                    self.vector_collection_text,
                    "text_embedding",
                    text_query_embedding,
                    k,
                )
                results_image = self._vector_search_in(
                    self.vector_collection_image,
                    "image_embedding",
                    text_query_embedding,
                    k,
                )
            else:
                # 단일
                results_text = self.vector_search(
                    text_query_embedding, limit=k, query_type="text"
                )
                results_image = self.vector_search(
                    text_query_embedding, limit=k, query_type="image"
                )

            # id 기준 가중 합산
            merged: Dict[str, Dict] = {}

            def _acc(lst: List[Dict], w: float):
                for item in lst:
                    pid = item.get("id")
                    if not pid:
                        continue
                    base = merged.get(pid, {"id": pid, "similarity_score": 0.0})
                    base["similarity_score"] += w * float(
                        item.get("similarity_score", 0.0)
                    )
                    merged[pid] = base

            _acc(results_text, wt)
            _acc(results_image, wi)

            # 상품 메타 배치 조인
            prod_col = self.client.collection(self.product_collection_name)
            refs = [prod_col.document(pid) for pid in merged.keys()]
            joined: Dict[str, Dict] = {}
            if refs:
                for pd in self.client.get_all(refs):  # ✅ 교정
                    if pd.exists:
                        joined[pd.id] = pd.to_dict()

            final = []
            for pid, row in merged.items():
                meta = joined.get(pid, {})
                final.append({**row, **meta})

            final.sort(key=lambda x: x.get("similarity_score", 0.0), reverse=True)
            return final[:limit]

        except Exception as e:
            self._nerror(f"❌ Cross-modal fusion search failed: {e}")
            return []

    # ----------------------
    # Math (남겨둠: 외부 유틸 dot_score 사용 중)
    # ----------------------
    def calculate_dot_product_score(
        self, vec1: List[float], vec2: List[float]
    ) -> float:
        try:
            vec1_np = np.array(vec1, dtype=np.float32)
            vec2_np = np.array(vec2, dtype=np.float32)
            return float(np.dot(vec1_np, vec2_np))
        except Exception as e:
            self.log.warning("Dot product calculation error: %s", e)
            return 0.0
