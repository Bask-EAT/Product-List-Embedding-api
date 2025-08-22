# Dockerized - Product-List-Embedding (Backend only)

이 프로젝트는 파이썬 기반 백엔드(추정: FastAPI/uvicorn)를 컨테이너에서 실행할 수 있도록 구성된 파일을 제공합니다.

## 포함 파일
- `Dockerfile.backend` — Python 3.11 slim 기반, `uvicorn`으로 앱 실행
- `docker-compose.yml` — 백엔드 단독 실행
- `.dockerignore` — 불필요 파일 제외
- `.env.example` — 샘플 환경 변수 (원 프로젝트 루트의 `.env` 사용을 기본값으로 설정)

## 사용법
```bash
# 1) 루트에 .env 가 이미 있습니다. 필요 시 내용 확인/수정
#    (또는 .env.example를 복사하여 .env 생성)

# 2) 빌드 & 실행
docker compose -f /mnt/data/dockerized/docker-compose.yml up -d --build

# 3) 확인
# Swagger UI (FastAPI일 경우): http://localhost:8000/docs
```

### 개발 모드 팁
- `docker-compose.yml`의 `volumes`를 주석 해제하면 로컬 코드를 컨테이너에 마운트하여
  빠른 수정-반영이 가능합니다. 이때 `uvicorn --reload`를 쓰도록 Dockerfile 또는 CMD를 조정하세요.

### 운영 모드 팁
- 로컬 마운트 대신 빌드 시 소스가 COPY 되는 현재 기본 방식을 권장합니다.
- Google/Firebase 자격 증명은 볼륨/비밀관리로 `/secrets/sa.json` 같은 경로에 마운트하고
  `GOOGLE_APPLICATION_CREDENTIALS=/secrets/sa.json` 환경변수로 지정하세요.
