import os
import torch
from PIL import Image
import streamlit as st
from multimodal_rag_system import MultimodalRAGSystem
from utils import validate_environment, display_search_results

from dotenv import load_dotenv

load_dotenv()


# GPU Setup for main info display
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    """Main Streamlit application"""

    st.set_page_config(
        page_title=os.getenv("PAGE_TITLE", "Multimodal RAG Search"),
        page_icon=os.getenv("PAGE_ICON", "🔍"),
        layout=os.getenv("LAYOUT", "wide"),
    )

    st.title("🔍 Multimodal RAG Search System")
    st.markdown(
        "**Search products using text, images, or both with Jina CLIP v2 + Firestore Vector DB**"
    )
    st.markdown("*Using dot product (inner product) similarity for ranking*")

    if not validate_environment():
        st.stop()

    if "rag_system" not in st.session_state:
        try:
            with st.spinner("🚀 Initializing RAG system..."):
                st.session_state.rag_system = MultimodalRAGSystem()
            st.success("✅ RAG system initialized successfully!")
        except Exception as e:
            st.error(f"❌ Failed to initialize RAG system: {e}")
            st.stop()

    rag_system = st.session_state.rag_system

    with st.sidebar:
        st.header("📊 데이터 관리")

        with st.expander("🔧 시스템 정보"):
            st.write(f"**Device:** {device}")
            st.write(f"**CUDA 사용 가능:** {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                st.write(f"**GPU:** {torch.cuda.get_device_name()}")
            st.write(f"**벡터 컬렉션:** {rag_system.vector_db.vector_collection_name}")
            st.write("**유사도 계산 방식:** Dot Product (내적)")
            st.write("**벡터 차원:** 1024")

    st.subheader("🔁 Firestore 조건 인덱싱")
    if st.button("🔁 Firestore에서 'is_emb=R' 상품 인덱싱"):
        with st.spinner("Embedding and storing products..."):
            if rag_system.vector_db.store_products(rag_system.embedding_model):
                st.success("✅ Firestore 상품 인덱싱 완료!")
            else:
                st.error("❌ Firestore 상품 인덱싱 실패")

    st.header("🔍 Search Interface")

    tab1, tab2, tab3 = st.tabs(
        ["📝 Text Search", "🖼️ Image Search", "🔀 Multimodal Search"]
    )

    with tab1:
        st.subheader("📝 Text Search")
        st.markdown("Search products using natural language queries")

        text_query = st.text_input(
            "Enter search query:",
            placeholder="예: 면류, 당면, 라면, 즉석식품",
            key="text_search_input",
        )

        col1, col2 = st.columns([3, 1])
        with col1:
            search_text_btn = st.button(
                "🔍 Search", key="text_search_btn", type="primary"
            )
        with col2:
            text_limit = st.selectbox(
                "Results", [30, 20, 10], index=0, key="text_limit"
            )

        if search_text_btn and text_query:
            with st.spinner("Searching..."):
                results = rag_system.search_by_text(text_query, text_limit)
                display_search_results(results)

    with tab2:
        st.subheader("🖼️ Image Search")
        st.markdown("Search products using uploaded images")

        uploaded_image = st.file_uploader(
            "Upload image:",
            type=["png", "jpg", "jpeg", "webp"],
            key="image_search_upload",
        )

        if uploaded_image:
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", width=300)

            col1, col2 = st.columns([3, 1])
            with col1:
                search_image_btn = st.button(
                    "🔍 Search", key="image_search_btn", type="primary"
                )
            with col2:
                image_limit = st.selectbox(
                    "Results", [30, 20, 10], index=0, key="image_limit"
                )

            if search_image_btn:
                with st.spinner("Processing image and searching..."):
                    results = rag_system.search_by_image(image, image_limit)
                    display_search_results(results)

    with tab3:
        st.subheader("🔀 Multimodal Search")
        st.markdown("Search products using both text and image for better precision")

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Text Query:**")
            multimodal_text = st.text_input(
                "Enter text query:",
                placeholder="예: 맛있는 면류",
                key="multimodal_text_input",
            )

        with col2:
            st.write("**Reference Image:**")
            multimodal_image = st.file_uploader(
                "Upload image:",
                type=["png", "jpg", "jpeg", "webp"],
                key="multimodal_image_upload",
            )

        if multimodal_image:
            image = Image.open(multimodal_image)
            st.image(image, caption="Reference Image", width=250)

        col1, col3, col2 = st.columns([3, 2, 1])
        with col1:
            search_multimodal_btn = st.button(
                "🔍 Search", key="multimodal_search_btn", type="primary"
            )
        with col2:
            multimodal_limit = st.selectbox(
                "Results", [30, 20, 10], index=0, key="multimodal_limit"
            )

        with col3:
            alpha = st.slider(
                "Alpha (Image Weight)",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.01,
                key="multimodal_alpha",
            )

        if search_multimodal_btn:
            if multimodal_text and multimodal_image:
                with st.spinner("Processing multimodal search..."):
                    image = Image.open(multimodal_image)
                    results = rag_system.search_multimodal(
                        multimodal_text, image, multimodal_limit, alpha
                    )
                    display_search_results(results)
            else:
                st.warning("⚠️ Please provide both text query and image")

    with st.expander("📋 Example Firestore Data Structure"):
        st.code(
            """
{
  "category": "Bakery_Jam",
  "id": "0000000010163",
  "image_url": "https://sitem.ssgcdn.com/63/01/01/item/0000000010163_i1_290.jpg",
  "is_emb": "D",
  "last_updated": "2025-08-06T15:44:42.153506",
  "out_of_stock": "N",
  "product_address": "https://emart.ssg.com/item/itemView.ssg?itemId=0000000010163&siteNo=6001&salestrNo=6005",
  "product_name": "[이팬트리] 스머커즈 구버 포도 땅콩버터잼 510g",
  "quantity": "100g 당 1,388원"
}

            """,
            language="json",
        )


if __name__ == "__main__":
    main()
