from typing import List, Dict
from PIL import Image
import streamlit as st
from jina_clip_embedding import JinaCLIPEmbedding
from firestore_vector_db import FirestoreVectorDB


class MultimodalRAGSystem:
    """Complete multimodal RAG system"""

    def __init__(self):
        self.embedding_model = JinaCLIPEmbedding()
        self.vector_db = FirestoreVectorDB()
        self.products_data = []

    def load_products_from_file(self, file_path: str) -> bool:
        """Load products from JSON file"""
        try:
            import json

            with open(file_path, "r", encoding="utf-8") as f:
                self.products_data = json.load(f)

            st.info(f"📁 Loaded {len(self.products_data)} products from {file_path}")
            return True

        except Exception as e:
            st.error(f"❌ Failed to load products: {e}")
            return False

    def index_products(self) -> bool:
        """Index products in vector database"""
        if not self.products_data:
            st.error("❌ No products loaded")
            return False

        return self.vector_db.store_products(self.embedding_model)

    def search_by_text(self, query: str, limit: int = 30) -> List[Dict]:
        """Search products using text query"""
        st.info(f"🔍 Text search: '{query}'")

        query_embedding = self.embedding_model.encode_text(
            query, task="retrieval.query"
        )
        if not query_embedding:
            return []

        results = self.vector_db.vector_search(query_embedding, limit)

        if results:
            product_id = [r.get("id", "Unknown") for r in results[:5]]
            st.info(f"✅ Top results: {', '.join(product_id)}")

        print(results)

        return results

    def search_by_image(
        self, image: Image.Image, limit: int = 30, query_type="image"
    ) -> List[Dict]:
        """Search products using image query"""
        st.info("🖼️ Processing image for search...")

        query_embedding = self.embedding_model.encode_image(image)
        if not query_embedding:
            return []

        results = self.vector_db.vector_search(query_embedding, limit, query_type)

        if results:
            product_names = [r.get("product_name", "Unknown") for r in results[:5]]
            st.info(f"✅ Top results: {', '.join(product_names)}")

        return results

    def search_multimodal(
        self, text_query: str, image: Image.Image, limit: int = 30, alpha: float = 0.7
    ) -> List[Dict]:
        st.info(f"🔀 Multimodal search: '{text_query}' + image")

        # 텍스트/이미지 임베딩 생성
        text_embedding = self.embedding_model.encode_text(
            text_query, task="retrieval.query"
        )
        image_embedding = self.embedding_model.encode_image(image)

        if not text_embedding or not image_embedding:
            return []

        # 벡터 DB에서 각각 검색
        text_results = self.vector_db.vector_search(
            text_embedding, limit=limit, query_type="text"
        )
        image_results = self.vector_db.vector_search(
            image_embedding, limit=limit, query_type="image"
        )

        # 중복 제거
        combined_dict = {}
        for doc in text_results + image_results:
            doc_id = doc.get("id")
            if doc_id and doc_id not in combined_dict:
                combined_dict[doc_id] = doc
        combined_results = list(combined_dict.values())

        # 각 문서의 텍스트/이미지 임베딩을 조합
        alpha_img, alpha_text = alpha, 1.0 - alpha
        for doc in combined_results:
            text_emb = doc.get("text_embedding", [])
            image_emb = doc.get("image_embedding", [])
            if text_emb and image_emb:
                doc["combined_embedding"] = [
                    alpha_img * i + alpha_text * t for i, t in zip(image_emb, text_emb)
                ]
            elif text_emb:
                doc["combined_embedding"] = text_emb
            elif image_emb:
                doc["combined_embedding"] = image_emb
            else:
                doc["combined_embedding"] = []

        # 질의 조합 임베딩 생성
        query_combined_embedding = [
            alpha_img * i + alpha_text * t
            for i, t in zip(image_embedding, text_embedding)
        ]

        # ✅ dot product 유사도 계산 함수
        def dot_product(a, b):
            return sum(x * y for x, y in zip(a, b))

        # ✅ dot product 기준 정렬
        combined_results.sort(
            key=lambda doc: dot_product(
                doc.get("combined_embedding", []), query_combined_embedding
            ),
            reverse=True,
        )

        top_results = combined_results[:limit]

        # 상위 결과 시각화
        if top_results:
            product_names = [r.get("product_name", "Unknown") for r in top_results[:5]]
            st.info(f"✅ Top results: {', '.join(product_names)}")

        return top_results
