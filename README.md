# 🔍 멀티모달 RAG 검색 시스템

Jina CLIP v2 임베딩과 구글 Firestore 벡터 DB를 이용해 **텍스트·이미지·멀티모달** 검색을
지원하는 스트림릿 애플리케이션입니다.

---

## 📂 프로젝트 구조
```
./
├── firestore_vector_db.py.     # Firestore 벡터 DB CRUD 및 검색
├── key/
│   └── key.json                # Firebase 키 파일
├── jina_clip_embedding.py.     # CLIP 모델 로드 및 임베딩
├── multimodal_rag_system.py.   # 텍스트·이미지·멀티모달 검색 로직
├── .env                        # 환경 변수 저장용 파일
├── requirements.txt            # 의존성 설치
├── utils.py.                   # 환경 변수 검증·검색 결과 UI
├── main.py.                    # Streamlit 실행 진입점
└── README.md                 
```

## 🛠️ 사전 준비
1. **Python 3.9+** 설치  
2. 가상환경 생성 및 활성화
   ```
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```
3. 의존성 설치
   ```
   pip install -r requirements.txt
   ```
4. `.env` 파일에 설정

## 🚀 실행 방법
```
streamlit run main.py
```

## 🖥️ UI 기능
| 탭 | 설명 |
| --- | --- |
| 📝 Text Search | 자연어 쿼리로 상품 검색 |
| 🖼️ Image Search | 업로드한 이미지와 유사한 상품 검색 |
| 🔀 Multimodal Search | 텍스트+이미지 동시 조건 검색 (α 슬라이더로 가중치 조절) |

## 🗄️ Firestore 스키마
- **상품 컬렉션** (예: `products`)  
  | 필드 | 타입 | 설명 |
  | --- | --- | --- |
  | id | string | 바코드·고유 ID |
  | product_name | string | 상품명 |
  | image_url | string | 상품 이미지 URL |
  | is_emb | string | 임베딩 상태 (`R` = 대기, `D` = 완료) |
  | ... | ... | 추가 메타데이터 |

- **벡터 컬렉션** (예: `product_vectors`)  
  | 필드 | 타입 | 설명 |
  | --- | --- | --- |
  | id | string | 상품 ID (FK) |
  | text_embedding | vector<float> | 1,024-차원 텍스트 임베딩 |
  | image_embedding | vector<float> | 1,024-차원 이미지 임베딩 |

## ⚙️ 주요 모듈 요약
1. `JinaCLIPEmbedding`  
   - 텍스트·이미지 임베딩 생성  
   - GPU 자동 감지 및 `bfloat16` 지원
2. `FirestoreVectorDB`  
   - 벡터 삽입·업데이트, Dot Product 유사도 기반 검색  
   - 상품 메타데이터 조인
3. `MultimodalRAGSystem`  
   - 텍스트, 이미지, 멀티모달 검색 인터페이스  
   - 멀티모달 시 α 값으로 가중 평균 후 코사인 유사도 정렬
4. `utils`  
   - 환경 변수 검증, 검색 결과 스트림릿 UI
5. `main`  
   - 스트림릿 페이지·사이드바·탭 구성  
   - 유저 입력 처리 및 결과 표시# Product-List-Embedding-api
