import os
import numpy as np
from typing import List, Dict
from google.cloud import firestore
from google.cloud.firestore_v1.vector import Vector
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure
from PIL import Image
from io import BytesIO
import requests
import streamlit as st

from dotenv import load_dotenv

load_dotenv()


class FirestoreVectorDB:
    """Firestore vector database client"""

    def __init__(self):
        self.client = None
        self.product_collection_name = os.getenv("FIRESTORE_PRODUCT_COLLECTION")
        self.vector_collection_name = os.getenv("FIRESTORE_VECTOR_COLLECTION")
        self.initialize_client()

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
            print(
                f"✅ Firestore initialized - Project: {project_id}\n"
                + f"Product Collection: {self.product_collection_name}\n"
                + f"Vector Collection: {self.vector_collection_name}\n"
            )

            self.test_connection()

        except Exception as e:
            st.error(f"❌ Firestore initialization failed: {e}")
            raise e

    def test_connection(self):
        """Test Firestore connection"""
        try:
            collections = list(self.client.collections())
            print(
                f"✅ Firestore connection verified - {len(collections)} collections found"
            )
        except Exception as e:
            st.error(f"❌ Firestore connection test failed: {e}")
            raise e

    def store_products(self, embedding_model):
        """Store products with embeddings in Firestore"""
        try:
            product_collection = self.client.collection(self.product_collection_name)
            embedding_collection = self.client.collection(self.vector_collection_name)

            query = product_collection.where("is_emb", "==", "R")
            docs = query.get()
            products_data = [doc.to_dict() for doc in docs]

            stored_count = 0

            progress_bar = st.progress(0)

            for i, product in enumerate(products_data):
                try:
                    embedding_text = embedding_model.encode_text(
                        product["product_name"]
                    )
                    image_url = product.get("image_url", "")
                except:
                    print(f"❌ embedding failed for product {product['id']}")
                    continue

                image = Image.open(BytesIO(requests.get(image_url).content)).convert(
                    "RGB"
                )
                embedding_image = embedding_model.encode_image(image)

                if embedding_text and embedding_image:
                    embedding_doc_data = {
                        "id": product["id"],
                        "text_embedding": Vector(embedding_text),
                        "image_embedding": Vector(embedding_image),
                    }

                    embedding_collection.document(product["id"]).set(embedding_doc_data)
                    product_collection.document(product["id"]).update({"is_emb": "D"})

                    stored_count += 1
                    print(f"✅ Stored: {product['product_name']}")

                progress_bar.progress((i + 1) / len(products_data))

            st.success(f"✅ Successfully stored {stored_count} products in Firestore")
            return True

        except Exception as e:
            st.error(f"❌ Failed to store products: {e}")
            return False

    def vector_search(
        self, query_embedding: List[float], limit: int = 30, query_type: str = "text"
    ) -> List[Dict]:
        """Perform vector similarity search using dot product"""
        try:
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

            results = []
            for doc in vector_query.stream():
                doc_data = doc.to_dict()
                embedding_vec = doc_data.get(vector_field_name)
                product_id = doc_data.get("id")

                if not isinstance(embedding_vec, list):
                    try:
                        embedding_vec = list(embedding_vec)
                    except:
                        embedding_vec = None

                similarity_score = 0
                if embedding_vec:
                    similarity_score = self.calculate_dot_product_score(
                        query_embedding, embedding_vec
                    )

                product_data = {}
                if product_id:
                    try:
                        product_doc = product_collection.document(product_id).get()
                        if product_doc.exists:
                            product_data = product_doc.to_dict()
                    except Exception as e:
                        print(f"❌ Failed to get product data for ID {product_id}: {e}")

                combined_result = {
                    "id": product_id,
                    "similarity_score": similarity_score,
                    "text_embedding": doc_data.get("text_embedding"),
                    "image_embedding": doc_data.get("image_embedding"),
                    **product_data,
                }

                results.append(combined_result)

            results.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
            return results

        except Exception as e:
            st.error(f"❌ Vector search failed: {e}")
            return []

    def calculate_dot_product_score(
        self, vec1: List[float], vec2: List[float]
    ) -> float:
        """Calculate dot product similarity score"""
        try:
            vec1_np = np.array(vec1)
            vec2_np = np.array(vec2)
            dot_product = np.dot(vec1_np, vec2_np)
            return float(dot_product)

        except Exception as e:
            print(f"❌ Dot product calculation error: {e}")
            return 0.0
