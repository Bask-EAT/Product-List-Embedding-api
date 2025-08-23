# ================================
# reset.ps1 - Clean rebuild script
# ================================

$ErrorActionPreference = "Stop"

# ---- 0) 사용자 환경 설정 ----
# 실제 서비스계정 키의 "윈도우 경로" (필요시 수정)
$KEY_SRC = "C:\Users\201\Desktop\final\Product-List-Embedding\keys\bask-eat-firebase-adminsdk-fbsvc-57cb4cbf5b.json"

# Docker Desktop 실행파일 경로 (기본)
$DockerDesktopExe = "C:\Program Files\Docker\Docker\Docker Desktop.exe"

Write-Host "==> STEP 0: 키 파일/도커 데스크탑 확인" -ForegroundColor Cyan

if (!(Test-Path $KEY_SRC)) {
  Write-Error "키 파일을 찾을 수 없습니다: $KEY_SRC"
}

# Docker Desktop 실행 여부 확인 (파이프 존재 검사)
function Test-DockerPipe {
  Test-Path "\\.\pipe\dockerDesktopLinuxEngine"
}

if (-not (Test-DockerPipe)) {
  Write-Host "Docker Desktop이 꺼져있습니다. 실행을 시도합니다..." -ForegroundColor Yellow
  Start-Process -FilePath $DockerDesktopExe | Out-Null

  # 파이프 준비 대기 (최대 120초)
  $maxWait = 120
  $elapsed = 0
  while (-not (Test-DockerPipe)) {
    Start-Sleep -Seconds 2
    $elapsed += 2
    if ($elapsed -ge $maxWait) {
      Write-Error "Docker Desktop 파이프가 준비되지 않았습니다. Docker Desktop을 켠 뒤 다시 실행하세요."
    }
  }
  Write-Host "Docker Desktop 준비 완료." -ForegroundColor Green
}

# docker 명령 가능 여부 확인
try {
  docker version | Out-Null
} catch {
  Write-Error "docker CLI 호출 실패: Docker Desktop을 켠 뒤 다시 실행하세요."
}

# ---- 1) Compose 내려서 정리 ----
Write-Host "==> STEP 1: compose down & prune" -ForegroundColor Cyan
try {
  docker compose down --rmi all --volumes --remove-orphans
} catch {
  Write-Host "compose down 중 오류(무시 가능): $($_.Exception.Message)" -ForegroundColor Yellow
}

# 강력 삭제 (이미지/네트워크/컨테이너/볼륨/빌드캐시)
docker system prune -a --volumes -f
docker builder prune --all --force

# ---- 2) 사전 검증: .env와 키 경로 확인 ----
Write-Host "==> STEP 2: .env / 경로 검증" -ForegroundColor Cyan

$envPath = ".\.env"
if (!(Test-Path $envPath)) { Write-Error ".env 파일이 없습니다. 먼저 .env를 생성하세요." }

# .env의 GOOGLE_APPLICATION_CREDENTIALS는 /keys/sa.json 이어야 함
$envText = Get-Content $envPath | Out-String
if ($envText -notmatch "GOOGLE_APPLICATION_CREDENTIALS\s*=\s*/keys/sa\.json") {
  Write-Error ".env의 GOOGLE_APPLICATION_CREDENTIALS는 '/keys/sa.json' 이어야 합니다."
}

# ---- 3) no-cache 빌드 ----
Write-Host "==> STEP 3: docker compose build --no-cache" -ForegroundColor Cyan
docker compose build --no-cache

# ---- 4) 키 파일 유효성 사전 검사 (컨테이너에서 일회성 실행) ----
Write-Host "==> STEP 4: 키 파일 유효성 검사 (컨테이너 내부)" -ForegroundColor Cyan
# 컨테이너 안에서 /keys/sa.json 이 존재하고 JSON으로 파싱되는지 체크
$checkCmd = @"
import os, json, sys
p = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "/keys/sa.json")
print("GOOGLE_APPLICATION_CREDENTIALS:", p)
if not os.path.isfile(p):
    print("exists: False")
    sys.exit(2)
print("exists: True")
try:
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    print("json: OK, type=", data.get("type"))
except Exception as e:
    print("json: INVALID ->", e)
    sys.exit(3)
"@

docker compose run --rm embed python -c $checkCmd
if ($LASTEXITCODE -ne 0) {
  Write-Error "키 파일 검증 실패: 바인드 경로나 JSON 형식을 확인하세요."
}

# ---- 5) 기동 ----
Write-Host "==> STEP 5: docker compose up -d" -ForegroundColor Cyan
docker compose up -d

# ---- 6) 런타임 확인 ----
Write-Host "==> STEP 6: 로그 팔로우 (Ctrl+C로 종료)" -ForegroundColor Cyan
docker compose logs -f embed
