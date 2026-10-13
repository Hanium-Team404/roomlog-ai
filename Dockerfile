# 1단계: texrecon(mvs-texturing) 빌드. R01 텍스처링에 쓰는 C++ 바이너리.
# 런타임 이미지에는 바이너리만 복사하므로 빌드 도구는 최종 이미지에 남지 않는다.
FROM python:3.11-slim AS texrecon-build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    ca-certificates \
    libpng-dev \
    libjpeg-dev \
    libtiff-dev \
    libtbb-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
# mvs-texturing과 하위 의존성(mve 등)이 구버전 cmake_minimum_required를 선언해 CMake 4.x에서 거부되므로
# 정책 하한을 지정한다 (CMake 3.x에서는 무시됨). 하위 ExternalProject에도 적용되게 환경변수로도 건다.
ENV CMAKE_POLICY_VERSION_MINIMUM=3.5
RUN git clone --recursive --depth 1 https://github.com/nmoehrle/mvs-texturing.git \
    && cd mvs-texturing && mkdir build && cd build \
    && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    && make -j"$(nproc)" \
    && test -x apps/texrecon/texrecon

# 2단계: 런타임
FROM python:3.11-slim

WORKDIR /app

ENV EGL_PLATFORM=surfaceless

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libegl1 \
    libgl1-mesa-dri \
    libosmesa6 \
    libpng16-16 \
    libjpeg62-turbo \
    libtiff6 \
    libtbb12 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=texrecon-build /src/mvs-texturing/build/apps/texrecon/texrecon /usr/local/bin/texrecon

RUN useradd -m -u 1000 appuser

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
