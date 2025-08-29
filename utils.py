import os
import streamlit as st
from typing import List, Dict

from dotenv import load_dotenv

load_dotenv()


def validate_environment() -> bool:
    """Validate required environment variables"""
    required_vars = ["GOOGLE_APPLICATION_CREDENTIALS", "GOOGLE_CLOUD_PROJECT"]

    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)

    if missing_vars:
        st.error(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        st.error("Please check your .env file")
        return False

    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not os.path.exists(creds_path):
        st.error(f"❌ Credentials file not found: {creds_path}")
        return False

    return True


def display_search_results(results: List[Dict]):
    """Display search results in Streamlit"""
    if not results:
        st.info("🔍 No results found")
        return

    st.subheader(f"📋 Found {len(results)} results (sorted by dot product score)")

    for i, result in enumerate(results):
        score = result.get("similarity_score", 0)
        product_name = result.get("product_name", "Unknown")

        with st.expander(f"#{i+1}: {product_name} (Score: {score:.4f})"):
            col1, col2 = st.columns([1, 2])

            with col1:
                if result.get("image_url"):
                    try:
                        st.image(
                            result["image_url"], width=200, caption="Product Image"
                        )
                    except:
                        st.text("📸 Image not available")

            with col2:
                st.write(f"**Category:** {result.get('category', 'N/A')}")
                st.write(f"**Product ID:** {result.get('id', 'N/A')}")
                st.write(
                    f"**Is Embedding:** {'Ready' if result.get('is_emb', 'R') == 'R' else 'Done'}"
                )
                st.write(f"**Last Updated:** {result.get('last_updated', 'N/A')}")
                st.write(f"**Product Name:** {result.get('product_name', 'N/A')}")
                st.write(f"**Quantity:** {result.get('quantity', 'N/A')}")
                st.write(
                    f"**In Stock:** {'✅ Yes' if result.get('out_of_stock', 'N') == 'N' else '❌ No'}"
                )
                st.write(f"**Dot Product Score:** {score:.4f}")

                if result.get("product_address"):
                    st.markdown(f"[🔗 View Product]({result['product_address']})")
