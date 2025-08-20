FROM python:3.10-slim AS wheel-builder

# Install build deps for wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc git wget curl libffi-dev libssl-dev python3-dev python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /wheels

# Copy requirements and prepare filtered requirements (remove torch packages)
COPY requirements.txt /tmp/requirements.txt
RUN python3 - <<'PY'
import pathlib
p=pathlib.Path('/tmp/requirements.txt')
lines=[l.strip() for l in p.read_text().splitlines() if l.strip() and not l.strip().startswith('#')]
filtered=[l for l in lines if not (l.startswith('torch') or l.startswith('torchvision') or l.startswith('torchaudio'))]
pathlib.Path('/tmp/req_filtered.txt').write_text('\n'.join(filtered))
print('Filtered requirements written:', '/tmp/req_filtered.txt')
PY

# Build wheels into /wheels/wheelhouse
RUN python3 -m pip install --upgrade pip setuptools wheel
RUN --mount=type=cache,target=/root/.cache/pip python3 -m pip wheel -r /tmp/req_filtered.txt --wheel-dir /wheels/wheelhouse

FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Install minimal runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 wget curl git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src

# Copy the prebuilt wheels from the wheel-builder stage
COPY --from=wheel-builder /wheels/wheelhouse /wheels/wheelhouse

# Copy requirements and filtered list
COPY requirements.txt /tmp/requirements.txt
RUN python3 - <<'PY'
import pathlib
p=pathlib.Path('/tmp/requirements.txt')
lines=[l.strip() for l in p.read_text().splitlines() if l.strip() and not l.strip().startswith('#')]
filtered=[l for l in lines if not (l.startswith('torch') or l.startswith('torchvision') or l.startswith('torchaudio'))]
pathlib.Path('/tmp/req_filtered.txt').write_text('\n'.join(filtered))
print('Filtered requirements written:', '/tmp/req_filtered.txt')
PY

# Install from local wheelhouse first, fallback to network if wheel missing
RUN python3 -m pip install --upgrade pip
RUN --mount=type=cache,target=/root/.cache/pip sh -c '\
    pip install --no-index --find-links=/wheels/wheelhouse -r /tmp/req_filtered.txt || \
    pip install --find-links=/wheels/wheelhouse -r /tmp/req_filtered.txt'

# Copy project files (models/ excluded via .dockerignore)
COPY . /src

# Add an entrypoint that downloads models at container start if missing
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# Default command: do nothing so cog can run its own entrypoint/command
CMD ["/bin/bash", "-c", "sleep infinity"]
