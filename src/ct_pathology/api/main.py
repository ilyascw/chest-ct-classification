"""FastAPI adapter for the CT pathology screening pipeline."""

from __future__ import annotations

import logging
import os
import shutil
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import cast

import torch
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from ct_pathology.pipeline.archive_utils import ArchiveValidationError
from ct_pathology.pipeline.core_pipeline import CTPathologyPipeline
from ct_pathology.pipeline.data_models import PipelineConfig

LOGGER = logging.getLogger(__name__)
DEFAULT_TMP_ROOT = Path("/tmp/ct_api")
DEFAULT_MAX_UPLOAD_MB = 10_240
UPLOAD_CHUNK_BYTES = 1024**2


def _env_positive_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer") from exc
    if value < 1:
        raise RuntimeError(f"{name} must be positive")
    return value


def _resolve_models_dir() -> Path:
    configured = os.getenv("CT_MODELS_DIR")
    candidates = [Path(configured)] if configured else [Path("/app/models"), Path("models")]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Models directory not found; searched: {searched}")


def build_pipeline() -> CTPathologyPipeline:
    """Builds the model pipeline from environment-backed configuration."""

    models_dir = _resolve_models_dir()
    ct_clip_path = models_dir / "CT_LiPro_v2.pt"
    catboost_path = models_dir / "catboost_pathology_classifier.cbm"
    for model_path in (ct_clip_path, catboost_path):
        if not model_path.is_file():
            raise FileNotFoundError(f"Required model not found: {model_path.name}")

    requested_device = os.getenv("CT_DEVICE", "auto").lower()
    if requested_device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = requested_device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CT_DEVICE=cuda requested, but CUDA is unavailable")

    config = PipelineConfig(
        ct_clip_checkpoint=str(ct_clip_path),
        catboost_model=str(catboost_path),
        device=device,
        max_workers=_env_positive_int("CT_MAX_WORKERS", 1),
        log_level=os.getenv("CT_LOG_LEVEL", "INFO"),
    )
    return CTPathologyPipeline(config)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    LOGGER.info("Initializing CT pathology pipeline")
    app.state.pipeline = await run_in_threadpool(build_pipeline)
    LOGGER.info("CT pathology pipeline is ready")
    try:
        yield
    finally:
        app.state.pipeline = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


app = FastAPI(
    title="CT Pathology Detection API",
    description="Research API for chest CT pathology screening",
    version="1.1.0",
    lifespan=lifespan,
)


def cleanup_temp_dir(temp_dir: Path) -> None:
    """Removes a completed request's temporary directory."""

    try:
        shutil.rmtree(temp_dir)
    except FileNotFoundError:
        return
    except OSError:
        LOGGER.exception("Failed to remove temporary directory %s", temp_dir)


async def _save_upload(
    upload: UploadFile,
    destination: Path,
    max_bytes: int,
) -> int:
    """Streams an upload to disk while enforcing a hard size limit."""

    written = 0
    with destination.open("wb") as output:
        while chunk := await upload.read(UPLOAD_CHUNK_BYTES):
            written += len(chunk)
            if written > max_bytes:
                raise HTTPException(
                    status_code=413,
                    detail="Uploaded archive exceeds the configured size limit",
                )
            output.write(chunk)
    return written


def _get_pipeline(request: Request) -> CTPathologyPipeline:
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline is not ready")
    return cast(CTPathologyPipeline, pipeline)


@app.get("/")  # type: ignore[misc]
async def root() -> dict[str, object]:
    return {
        "status": "running",
        "service": "CT Pathology Detection API",
        "version": app.version,
        "docs": "/docs",
    }


@app.get("/health")  # type: ignore[misc]
async def health(request: Request) -> dict[str, object]:
    pipeline_ready = getattr(request.app.state, "pipeline", None) is not None
    return {
        "status": "healthy" if pipeline_ready else "starting",
        "pipeline_ready": pipeline_ready,
        "gpu_available": torch.cuda.is_available(),
        "models_loaded": pipeline_ready,
    }


@app.post("/predict")  # type: ignore[misc]
async def predict(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
) -> FileResponse:
    """Processes a ZIP archive and returns an XLSX report."""

    pipeline = _get_pipeline(request)
    original_name = file.filename or ""
    safe_name = Path(original_name).name
    if not safe_name or not safe_name.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only ZIP files are supported")

    job_id = uuid.uuid4().hex[:12]
    temp_root = Path(os.getenv("CT_API_TMP_DIR", str(DEFAULT_TMP_ROOT)))
    temp_dir = temp_root / job_id
    temp_dir.mkdir(parents=True, exist_ok=False)
    zip_path = temp_dir / safe_name
    output_excel = temp_dir / f"results_{job_id}.xlsx"
    max_upload_bytes = (
        _env_positive_int(
            "CT_MAX_UPLOAD_MB",
            DEFAULT_MAX_UPLOAD_MB,
        )
        * 1024**2
    )

    try:
        upload_size = await _save_upload(file, zip_path, max_upload_bytes)
        LOGGER.info(
            "Processing job %s: archive=%s size_bytes=%d",
            job_id,
            safe_name,
            upload_size,
        )

        await run_in_threadpool(
            pipeline.process_zip_archives,
            [str(zip_path)],
            str(output_excel),
        )
        if not output_excel.is_file():
            raise RuntimeError("Pipeline did not create an output report")

        background_tasks.add_task(cleanup_temp_dir, temp_dir)
        return FileResponse(
            path=output_excel,
            filename=f"ct_pathology_results_{job_id}.xlsx",
            media_type=("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
        )
    except HTTPException:
        cleanup_temp_dir(temp_dir)
        raise
    except ArchiveValidationError as exc:
        cleanup_temp_dir(temp_dir)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        cleanup_temp_dir(temp_dir)
        LOGGER.exception("Job %s failed", job_id)
        raise HTTPException(
            status_code=500,
            detail="CT processing failed; inspect server logs using the job ID",
            headers={"X-Job-ID": job_id},
        ) from exc
    finally:
        await file.close()


@app.get("/info")  # type: ignore[misc]
async def info(request: Request) -> dict[str, object]:
    pipeline = _get_pipeline(request)
    return {
        "ct_clip_checkpoint": Path(pipeline.config.ct_clip_checkpoint).name,
        "catboost_model": Path(pipeline.config.catboost_model).name,
        "device": pipeline.config.device,
        "max_workers": pipeline.config.max_workers,
    }
