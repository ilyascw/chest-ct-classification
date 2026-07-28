# Docker deployment

The image targets Linux hosts with an NVIDIA GPU, a recent driver compatible
with CUDA 12.8, Docker Engine and NVIDIA Container Toolkit.

Before building, place both required artifacts in `models/`; see
[`models/README.md`](models/README.md).

```bash
cp .env.example .env
docker compose build
docker compose up -d
docker compose logs -f ct-pathology-api
```

Readiness and API schema:

```bash
curl --fail http://localhost:8000/health
```

OpenAPI UI is available at `http://localhost:8000/docs`.

Run one request:

```bash
curl --fail-with-body \
  -F "file=@study.zip" \
  http://localhost:8000/predict \
  --output results.xlsx
```

The service uses one Uvicorn worker because the CT-CLIP model is GPU-resident.
`CT_MAX_WORKERS` controls study-level processing within that process. Uploaded
archives are streamed to `runtime_tmp/`, validated before extraction, and
removed after the response is sent.

Stop the service with `docker compose down`. The model files are baked into the
local image; `runtime_tmp/` and `hf_cache/` are ignored by Git.
