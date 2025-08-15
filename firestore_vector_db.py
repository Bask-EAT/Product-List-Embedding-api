# firestore_vector_db.py
from __future__ import annotations

import os
import logging
import numpy as np
import requests
from io import BytesIO
from typing import List, Dict, Optional, Callable
from collections import defaultdict  # ✅ 추가

from PIL import Image
from dotenv import load_dotenv

from google.cloud import firestore
from google.cloud.firestore_v1 import FieldFilter  # ✅ 신식 필터
from google.cloud.firestore_v1.vector import Vector
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure

load_dotenv()


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
        self.vector_collection_name = os.getenv("FIRESTORE_VECTOR_COLLECTION")
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
        """Initialize Firestore client with authentication"""
        try:
            project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
            creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

            if not project_id:
                raise ValueError("GOOGLE_CLOUD_PROJECT environment variable not set")

            if not creds_path or not os.path.exists(creds_path):
                raise ValueError(f"Invalid credentials file: {creds_path}")

            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path

            self.client = firestore.Client(project=project_id)
            self._ninfo(
                "✅ Firestore initialized - "
                f"Project: {project_id}\n"
                f"Product Collection: {self.product_collection_name}\n"
                f"Vector Collection: {self.vector_collection_name}"
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
        text_task: str = "retrieval.document",  # 필요 시 조정
        image_timeout_sec: int = 10,
    ) -> bool:
        """Store products with embeddings in Firestore"""
        try:
            assert self.client is not None
            product_collection = self.client.collection(self.product_collection_name)
            embedding_collection = self.client.collection(self.vector_collection_name)

            # ✅ 신식 필터 API 사용
            query = product_collection.where(filter=FieldFilter("is_emb", "==", "R"))
            docs = query.get()
            products_data = [doc.to_dict() for doc in docs]

            total = len(products_data)
            if total == 0:
                self._ninfo("ℹ️ No products to index (is_emb == 'R') ")
                if progress_cb:
                    progress_cb(1.0)
                return True

            stored_count = 0
            for i, product in enumerate(products_data):
                try:
                    # ---- Text embedding
                    product_name = product.get("product_name") or ""
                    if not product_name:
                        raise ValueError("product_name is empty")

                    embedding_text = embedding_model.encode_text(
                        product_name, task=text_task
                    )
                    if not embedding_text:
                        raise RuntimeError("text embedding failed")

                    # ---- Image embedding
                    image_url = product.get("image_url") or ""
                    if not image_url:
                        raise ValueError("image_url is empty")

                    resp = requests.get(image_url, timeout=image_timeout_sec)
                    resp.raise_for_status()
                    image = Image.open(BytesIO(resp.content)).convert("RGB")

                    embedding_image = embedding_model.encode_image(image)
                    if not embedding_image:
                        raise RuntimeError("image embedding failed")

                    # ---- Write embeddings
                    pid = product.get("id")
                    if not pid:
                        raise ValueError("product id is missing")

                    embedding_doc_data = {
                        "id": pid,
                        "text_embedding": Vector(embedding_text),
                        "image_embedding": Vector(embedding_image),
                    }

                    embedding_collection.document(pid).set(embedding_doc_data)
                    product_collection.document(pid).update({"is_emb": "D"})

                    stored_count += 1
                    self.log.debug("Stored embeddings for product id=%s", pid)

                except Exception as ex:
                    self.log.exception("Embedding/store failed for product: %s", ex)

                # progress
                if progress_cb:
                    progress_cb((i + 1) / total)

            self._ninfo(f"✅ Successfully stored {stored_count}/{total} products")
            return True

        except Exception as e:
            self._nerror(f"❌ Failed to store products: {e}")
            return False

    # ----------------------
    # Aggregation (NEW)
    # ----------------------
    def get_category_counts(
        self,
        only_embedded: Optional[
            bool
        ] = None,  # True: is_emb == "D", False: "R", None: 전체
        category_field: str = "category",  # 카테고리 필드명
    ) -> Dict[str, int]:
        """
        카테고리별 문서 수를 집계합니다.
        반환 예: {"Fruits": 120, "Vegetables": 87, ...}
        """
        assert self.client is not None
        col = self.client.collection(self.product_collection_name)

        # is_emb 조건 적용
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

        # 라벨 알파벳순 정렬된 dict로 반환
        return dict(sorted(counts.items(), key=lambda kv: kv[0].lower()))

    def get_category_distribution(
        self,
        only_embedded: Optional[bool] = None,
        category_field: str = "category",
    ) -> Dict[str, object]:
        """
        카테고리별 개수/총합/비율을 함께 반환합니다.
        예: {"counts": {...}, "total": 1234, "ratios": {"Fruits": 0.23, ...}}
        """
        counts = self.get_category_counts(
            only_embedded=only_embedded, category_field=category_field
        )
        total = sum(counts.values())
        ratios = {k: (v / total if total else 0.0) for k, v in counts.items()}
        return {"counts": counts, "total": total, "ratios": ratios}

    # ----------------------
    # Vector search
    # ----------------------
    def vector_search(
        self, query_embedding: List[float], limit: int = 30, query_type: str = "text"
    ) -> List[Dict]:
        """Perform vector similarity search using dot product"""
        try:
            assert self.client is not None
            collection = self.client.collection(self.vector_collection_name)
            product_collection = self.client.collection(self.product_collection_name)

            query_vector = Vector(query_embedding)
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
            for doc in vector_query.stream():
                doc_data = doc.to_dict()
                product_id = doc_data.get("id")

                # 임베딩 벡터는 Vector 또는 list일 수 있음
                embedding_vec = doc_data.get(vector_field_name)
                if not isinstance(embedding_vec, list):
                    try:
                        embedding_vec = list(embedding_vec)  # Vector -> list
                    except Exception:
                        embedding_vec = None

                # dot product 유사도 계산 (큰 값이 더 유사)
                similarity_score = 0.0
                if embedding_vec:
                    similarity_score = self.calculate_dot_product_score(
                        query_embedding, embedding_vec
                    )

                # 원본 상품 정보 조인
                product_data: Dict = {}
                if product_id:
                    try:
                        product_doc = product_collection.document(product_id).get()
                        if product_doc.exists:
                            product_data = product_doc.to_dict()
                    except Exception as ex:
                        self.log.warning(
                            "Failed to load product doc id=%s: %s", product_id, ex
                        )

                combined_result = {
                    "id": product_id,
                    "similarity_score": similarity_score,
                    "text_embedding": doc_data.get("text_embedding"),
                    "image_embedding": doc_data.get("image_embedding"),
                    **product_data,
                }
                results.append(combined_result)

            # 안전하게 한 번 더 정렬
            results.sort(key=lambda x: x.get("similarity_score", 0.0), reverse=True)
            return results

        except Exception as e:
            self._nerror(f"❌ Vector search failed: {e}")
            return []

    # ----------------------
    # Math
    # ----------------------
    def calculate_dot_product_score(
        self, vec1: List[float], vec2: List[float]
    ) -> float:
        """Calculate dot product similarity score"""
        try:
            vec1_np = np.array(vec1, dtype=np.float32)
            vec2_np = np.array(vec2, dtype=np.float32)
            return float(np.dot(vec1_np, vec2_np))
        except Exception as e:
            self.log.warning("Dot product calculation error: %s", e)
            return 0.0
