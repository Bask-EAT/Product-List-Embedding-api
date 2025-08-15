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

        # 기존 구조 유지: FirestoreVectorDB가 내부에서 products_data를 참조한다면
        # 그대로 두세요. 필요 시 아래처럼 전달하는 방식으로 변경하세요.
        # return self.vector_db.store_products(self.embedding_model, self.products_data)
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
    def search_by_text(self, query: str, limit: int = 30) -> List[Dict]:
        """Search products using text query"""
        self._ninfo(f"🔍 Text search: '{query}'")

        query_embedding = self.embedding_model.encode_text(
            query, task="retrieval.query"
        )
        if not query_embedding:
            self._nerror("❌ Failed to create text embedding")
            return []

        results = self.vector_db.vector_search(query_embedding, limit)
        if results:
            product_ids = [r.get("id", "Unknown") for r in results[:5]]
            self._ninfo(f"✅ Top results: {', '.join(product_ids)}")

        # self.log.debug("search_by_text results: %s", results)  # 원하면 디버그 로그
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
        Multimodal search: combine text & image signals.
        alpha: weight for image (0~1), text weight = 1 - alpha
        """
        self._ninfo(f"🔀 Multimodal search: '{text_query}' + image")

        text_embedding = self.embedding_model.encode_text(
            text_query, task="retrieval.query"
        )
        image_embedding = self.embedding_model.encode_image(image)

        if not text_embedding or not image_embedding:
            self._nerror("❌ Failed to create multimodal embeddings")
            return []

        # Independent searches
        text_results = (
            self.vector_db.vector_search(text_embedding, limit=limit, query_type="text")
            or []
        )
        image_results = (
            self.vector_db.vector_search(
                image_embedding, limit=limit, query_type="image"
            )
            or []
        )

        # Deduplicate by doc id
        combined_dict: Dict[str, Dict] = {}
        for doc in text_results + image_results:
            doc_id = doc.get("id")
            if doc_id and doc_id not in combined_dict:
                combined_dict[doc_id] = doc
        combined_results: List[Dict] = list(combined_dict.values())

        # Build combined embedding per doc
        alpha_img, alpha_text = alpha, 1.0 - alpha
        for doc in combined_results:
            text_emb = doc.get("text_embedding", [])
            image_emb = doc.get("image_embedding", [])
            if text_emb and image_emb:
                # zip: 길이 불일치 시 최소 길이에 맞춰 자름(모델 차원 동일 가정)
                doc["combined_embedding"] = [
                    alpha_img * i + alpha_text * t for i, t in zip(image_emb, text_emb)
                ]
            elif text_emb:
                doc["combined_embedding"] = text_emb
            elif image_emb:
                doc["combined_embedding"] = image_emb
            else:
                doc["combined_embedding"] = []

        # Query combined embedding
        query_combined_embedding = [
            alpha_img * i + alpha_text * t
            for i, t in zip(image_embedding, text_embedding)
        ]

        # Dot product similarity
        def dot_product(a, b):
            return sum(x * y for x, y in zip(a, b))

        combined_results.sort(
            key=lambda d: dot_product(
                d.get("combined_embedding", []), query_combined_embedding
            ),
            reverse=True,
        )

        top_results = combined_results[:limit] if combined_results else []

        if top_results:
            product_names = [r.get("product_name", "Unknown") for r in top_results[:5]]
            self._ninfo(f"✅ Top results: {', '.join(product_names)}")

        return top_results
