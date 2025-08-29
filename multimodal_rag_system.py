# multimodal_rag_system.py
from __future__ import annotations
from typing import List, Dict, Optional, Callable
from PIL import Image
import logging

from jina_clip_embedding import JinaCLIPEmbedding
from firestore_vector_db import FirestoreVectorDB


class MultimodalRAGSystem:
    """
    Complete multimodal RAG system (backend-safe)
    - No hard dependency on Streamlit.
    - Optional notifier callable for UI environments (e.g., Streamlit).
    """

    def __init__(
        self,
        notify: Optional[Callable[[str, str], None]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        # notify(level, message): "info" | "warning" | "error"
        self._notify = notify or (lambda level, msg: None)
        self.log = logger or logging.getLogger("multimodal_rag")

        self.embedding_model = JinaCLIPEmbedding()
        self.vector_db = FirestoreVectorDB()
        self.products_data: List[Dict] = []

    def _ninfo(self, msg: str):
        self.log.info(msg)
        self._notify("info", msg)

    def _nerror(self, msg: str):
        self.log.error(msg)
        self._notify("error", msg)

    # -------------------------
    # Data loading / indexing
    # -------------------------
    def load_products_from_file(self, file_path: str) -> bool:
        """Load products from JSON file"""
        try:
            import json

            with open(file_path, "r", encoding="utf-8") as f:
                self.products_data = json.load(f)

            self._ninfo(
                f"📁 Loaded {len(self.products_data)} products from {file_path}"
            )
            return True

        except Exception as e:
            self._nerror(f"❌ Failed to load products: {e}")
            return False

    def index_products(self) -> bool:
        """Index products in vector database"""
        if not self.products_data:
            self._nerror("❌ No products loaded")
            return False

        try:
            ok = self.vector_db.store_products(self.embedding_model)
            if ok:
                self._ninfo("✅ Products indexed successfully")
            else:
                self._nerror("❌ Failed to index products")
            return ok
        except Exception as e:
            self._nerror(f"❌ Indexing error: {e}")
            return False

    # -------------------------
    # Search: text / image / multimodal
    # -------------------------
    def search_by_text(
        self,
        query: str,
        limit: int = 30,
        use_crossmodal: bool = False,
        text_weight: float = 0.6,
        image_weight: float = 0.4,
    ) -> List[Dict]:
        """
        Search products using text query.
        - use_crossmodal=True: 텍스트 쿼리 하나로 텍스트+이미지 융합(Late Fusion) 검색 사용
        """
        if use_crossmodal:
            return self.search_by_text_crossmodal(
                query=query,
                limit=limit,
                text_weight=text_weight,
                image_weight=image_weight,
            )

        self._ninfo(f"🔍 Text search: '{query}'")

        query_embedding = self.embedding_model.encode_text(
            query, task="retrieval.query"
        )
        if not query_embedding:
            self._nerror("❌ Failed to create text embedding")
            return []

        results = self.vector_db.vector_search(
            query_embedding, limit, query_type="text"
        )
        if results:
            product_ids = [r.get("id", "Unknown") for r in results[:5]]
            self._ninfo(f"✅ Top results: {', '.join(product_ids)}")

        return results or []

    def search_by_text_crossmodal(
        self,
        query: str,
        limit: int = 30,
        text_weight: float = 0.6,
        image_weight: float = 0.4,
    ) -> List[Dict]:
        """
        텍스트 쿼리 1개로 '텍스트 임베딩'과 '이미지 임베딩' 컬럼을 각각 검색한 뒤,
        가중치 기반 Late Fusion으로 최종 랭킹을 반환합니다.
        (firestore_vector_db.vector_search_crossmodal_text 사용)
        """
        self._ninfo(
            f"🔀 Crossmodal (Late Fusion) text search: '{query}' "
            f"(w_text={text_weight:.2f}, w_image={image_weight:.2f})"
        )

        query_embedding = self.embedding_model.encode_text(
            query, task="retrieval.query"
        )
        if not query_embedding:
            self._nerror("❌ Failed to create text embedding for crossmodal search")
            return []

        results = self.vector_db.vector_search_crossmodal_text(
            text_query_embedding=query_embedding,
            limit=limit,
            weights=(text_weight, image_weight),
        )

        if results:
            names = [r.get("product_name", r.get("id", "Unknown")) for r in results[:5]]
            self._ninfo(f"✅ Crossmodal top results: {', '.join(names)}")

        return results or []

    def search_by_image(
        self, image: Image.Image, limit: int = 30, query_type: str = "image"
    ) -> List[Dict]:
        """Search products using image query"""
        self._ninfo("🖼️ Processing image for search...")

        query_embedding = self.embedding_model.encode_image(image)
        if not query_embedding:
            self._nerror("❌ Failed to create image embedding")
            return []

        results = self.vector_db.vector_search(query_embedding, limit, query_type)
        if results:
            product_names = [r.get("product_name", "Unknown") for r in results[:5]]
            self._ninfo(f"✅ Top results: {', '.join(product_names)}")

        return results or []

    def search_multimodal(
        self, text_query: str, image: Image.Image, limit: int = 30, alpha: float = 0.7
    ) -> List[Dict]:
        """
        Multimodal search (Late Fusion):
        - 텍스트/이미지 각각 별도 검색 → alpha 가중치로 점수 합산 → 최종 랭킹
        - 백엔드(vector_search 응답에 임베딩 미포함)와 완전 호환
        """
        self._ninfo(
            f"🔀 Multimodal search (Late Fusion): '{text_query}' + image (alpha={alpha:.2f})"
        )

        # 쿼리 임베딩
        text_emb = self.embedding_model.encode_text(text_query, task="retrieval.query")
        image_emb = self.embedding_model.encode_image(image)
        if not text_emb or not image_emb:
            self._nerror("❌ Failed to create multimodal embeddings")
            return []

        # 개별 검색 (융합 전 넉넉히 가져오기)
        K = max(limit * 2, 50)
        text_res = (
            self.vector_db.vector_search(text_emb, limit=K, query_type="text") or []
        )
        image_res = (
            self.vector_db.vector_search(image_emb, limit=K, query_type="image") or []
        )

        # id 기준 가중 합산
        w_img, w_text = alpha, (1.0 - alpha)
        merged: Dict[str, Dict] = {}

        def _acc(lst, w):
            for it in lst:
                pid = it.get("id")
                if not pid:
                    continue
                base = merged.get(pid, {"id": pid, "similarity_score": 0.0})
                base["similarity_score"] += w * float(it.get("similarity_score", 0.0))
                # 메타가 비어있으면 한 번만 채운다
                if "product_name" not in base and isinstance(it, dict):
                    for k, v in it.items():
                        if k != "similarity_score" and k not in base:
                            base[k] = v
                merged[pid] = base

        _acc(text_res, w_text)
        _acc(image_res, w_img)

        out = list(merged.values())
        out.sort(key=lambda x: x.get("similarity_score", 0.0), reverse=True)
        top = out[:limit]

        if top:
            names = [r.get("product_name", r.get("id", "Unknown")) for r in top[:5]]
            self._ninfo(f"✅ Top results: {', '.join(names)}")
        return top
