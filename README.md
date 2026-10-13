# roomlog-ai

> 자취/원룸 거주자를 위한 **3D 방 기록 & 하자 관리 서비스**의 AI 서버
> 2026 한이음 드림업 프로젝트 (Team 404)

LiDAR로 촬영한 방을 3D로 재구성하고, GPT Vision + SAM3으로 하자를 자동 탐지해 3D 좌표까지 매핑하는 서비스의 FastAPI AI 서버입니다.

## ✨ 주요 기능

| 기능 | 설명 |
|------|------|
| **3D 재구성 (R01)** | LiDAR 원시 데이터(.zip)를 받아 pose 보정, TSDF 메시, 원본 키프레임 텍스처링(mvs-texturing)을 거쳐 텍스처 usdz 한 개를 만들어 S3에 저장합니다. 썸네일(4뷰 렌더 + 합성 이미지)도 함께 생성합니다. |
| **하자 탐지 (D01)** | 스캔 영상에서 프레임을 추출해 GPT Vision으로 하자를 탐지하고, SAM3 세그멘테이션으로 정밀한 마스크를 구합니다. depth map과 odometry를 역투영해 각 하자의 3D 위치를 산출합니다. |
| **하자 비교 (D02)** | 입주·퇴거 시 두 스캔의 하자 목록을 비교해 새로 생긴 하자만 추출합니다. 3D 무게중심 거리(0.3 m 임계값)로 기존 하자와 신규 하자를 구분합니다. |

## 🚀 AI 서버 포인트

### 비동기 처리 파이프라인

3D 재구성과 하자 분석은 수십 초 이상 걸려 동기 응답이 불가합니다. 모든 작업은 `BackgroundTasks`로 즉시 202를 반환하고, 완료 후 백엔드 콜백 URL로 결과를 전송합니다.

```
POST /reconstruction        POST /defect-detection        POST /defect-comparison
        │                           │                              │
  202 즉시 반환              202 즉시 반환                  202 즉시 반환
        │                           │                              │
  [BackgroundTask]           [BackgroundTask]               [BackgroundTask]
  ZIP 다운로드                ZIP 다운로드                  ZIP 다운로드 (필요 시)
  pose 보정 + TSDF            프레임 추출                    D01 병렬 실행
  texrecon 텍스처링 → usdz    GPT Vision 탐지                3D 중심점 비교
  usdz S3 업로드, 썸네일      SAM3 세그멘테이션              신규 하자 추출
        │                   depth → 3D 역투영                      │
        └──────────────────────── callback_url POST ───────────────┘
```

### 하자 탐지 파이프라인 (D01)

GPT Vision이 영상 프레임에서 하자의 종류·심각도·위치·bbox를 JSON으로 반환하면, SAM3가 해당 bbox를 기반으로 픽셀 수준 마스크를 생성합니다. 이후 depth map과 camera matrix, odometry로 2D 폴리곤을 3D 공간 좌표로 역투영합니다.

```
영상 프레임 추출 (30프레임마다, 최대 20장)
        ↓
GPT Vision → 하자 bbox + 메타데이터 (JSON)
        ↓
fal.ai SAM3 → 픽셀 마스크 → 2D 폴리곤
        ↓
depth map × camera matrix × odometry → 3D 좌표 (region_3d)
        ↓
S3 업로드 (bbox 시각화 / 마스크 시각화 / 1:1.6 크롭)
```

### 입주·퇴거 비교 (D02)

기존 D01 결과 JSON이 있으면 재처리 없이 재사용하고, 없으면 D01을 병렬 실행합니다. 두 결과의 하자 무게중심 거리가 0.3 m 미만이면 기존 하자로 판단해 제외합니다.

```
입주 하자 목록 (JSON 재사용 또는 D01 신규 실행)
퇴거 하자 목록 (JSON 재사용 또는 D01 신규 실행)
        ↓
3D 무게중심 거리 비교 (임계값 0.3 m)
        ↓
신규 하자만 콜백 전송
```

## 🛠 기술 스택

- **언어/프레임워크**: Python 3.11, FastAPI
- **AI 모델**: OpenAI GPT Vision (하자 탐지), fal.ai SAM3 (세그멘테이션)
- **3D 재구성**: Open3D (Scalable TSDF Volume, ICP pose graph), mvs-texturing(texrecon), trimesh, usd-core, SciPy, NumPy
- **영상 처리**: OpenCV, Pillow
- **파일 저장**: AWS S3 호환 (boto3)
- **컨테이너**: Docker (EGL surfaceless + OSMesa 헤드리스 렌더링)
- **문서**: FastAPI 내장 Swagger UI (`/docs`)

## 🏛 패키지 구조

```text
app
├── main.py                  — FastAPI 앱 초기화, API Key 미들웨어
├── routers
│   ├── reconstruction.py    — AI-R01. 3D 재구성(usdz) 엔드포인트
│   ├── defect_detection.py  — AI-D01. 하자 탐지 엔드포인트
│   └── defect_comparison.py — AI-D02. 입주/퇴거 하자 비교 엔드포인트
├── services
│   ├── reconstruction/      — R01 파이프라인 (preprocess → tsdf → texture → usdz)
│   │   ├── preprocess.py    — RGB 동기화, depth 필터, loop closure pose 보정, 키프레임 선택
│   │   ├── tsdf.py          — TSDF 융합, cleanup, 텍스처용 단순화
│   │   ├── texture.py       — texrecon 씬 구성, 좌표계 검증, 텍스처링
│   │   ├── usdz.py          — obj → usdz (미관측 면은 TSDF 정점 색으로 굽기)
│   │   └── pipeline.py      — 단계 연결
│   ├── vision_service.py    — GPT Vision 탐지 + SAM3 세그멘테이션 + 3D 역투영
│   ├── frame_service.py     — 영상 프레임 추출
│   ├── pose_service.py      — 2D 폴리곤 → 3D 좌표 변환 (역투영)
│   └── thumbnail_service.py — 메시 렌더링 및 썸네일 생성
├── core
│   ├── config.py            — 환경 변수 설정 (pydantic-settings)
│   ├── callback.py          — 백엔드 콜백 POST 클라이언트 (httpx)
│   ├── storage.py           — S3 업로드 / 스캔 ZIP 다운로드
│   └── redis_client.py      — Redis 클라이언트
├── models
│   ├── request.py           — API 요청 스키마 (ReconstructionRequest 등)
│   └── response.py          — API 응답 스키마 (DefectItem, Point3D 등)
└── prompts
    └── defect_detection.txt — GPT Vision 시스템 프롬프트
```

## ⚙️ 환경 변수

`.env` 파일 또는 환경 변수로 설정합니다.

| 변수 | 설명 | 기본값 |
|------|------|--------|
| `OPENAI_API_KEY` | OpenAI API 키 (GPT Vision) | 필수 |
| `FAL_API_KEY` | fal.ai API 키 (SAM3) | 필수 |
| `S3_BUCKET_NAME` | S3 버킷 이름 | 필수 |
| `S3_REGION` | S3 리전 | `ap-northeast-2` |
| `AWS_ACCESS_KEY_ID` | AWS 액세스 키 | 선택 |
| `AWS_SECRET_ACCESS_KEY` | AWS 시크릿 키 | 선택 |
| `API_KEY` | 내부 API 인증 키 (`X-Api-Key` 헤더) | `roomlog-ai-secret-key` |
| `REDIS_URL` | Redis 연결 URL | `redis://localhost:6379` |
| `RECON_FRAME_STRIDE` | R01 텍스처 소스 프레임 간격 (작을수록 커버리지↑ 시간·메모리↑) | `9` |
| `RECON_TARGET_FACES` | R01 텍스처링용 메시 삼각형 수 | `150000` |
| `RECON_REFINE_POSES` | R01 loop closure pose 보정 (D01 하자 좌표도 같은 보정 사용) | `true` |
| `RECON_NUM_THREADS` | texrecon 스레드 수 | 전부 |
| `TEXRECON_BIN` | texrecon 바이너리 경로 | `texrecon` |

## 🐳 실행

```bash
# 개발 서버
uvicorn app.main:app --reload --port 8000

# Docker
docker build -t roomlog-ai .
docker run -p 8000:8000 --env-file .env roomlog-ai
```

## 📄 License

[MIT License](LICENSE) © 2026 Team404
