from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import shutil
from pathlib import Path
import uuid
import sys
import os

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.pipeline.core_pipeline import CTPathologyPipeline
from src.pipeline.data_models import PipelineConfig

app = FastAPI(title="CT Pathology API")

pipeline = None

@app.on_event("startup")
async def startup():
    global pipeline
    
    # В контейнере модели будут в /app/models/
    base_path = Path("/app/models")
    
    config = PipelineConfig(
        ct_clip_checkpoint=str(base_path / "CT_LiPro_v2.pt"),
        catboost_model=str(base_path / "catboost_pathology_classifier.cbm"),
        device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",  # Автодетект
        max_workers=2,
        log_level="INFO"
    )
    pipeline = CTPathologyPipeline(config)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())[:8]
    temp_dir = Path("/tmp/ct_api")
    temp_dir.mkdir(exist_ok=True)
    
    temp_zip = temp_dir / f"{job_id}.zip"
    
    with temp_zip.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    
    output_excel = temp_dir / f"{job_id}_results.xlsx"
    
    try:
        pipeline.process_zip_archives([str(temp_zip)], str(output_excel))
        return FileResponse(
            output_excel, 
            filename="results.xlsx",
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    finally:
        temp_zip.unlink(missing_ok=True)  # Очищаем

@app.get("/health")
async def health():
    return {"status": "ok", "pipeline_ready": pipeline is not None}

@app.get("/")
async def root():
    return {"message": "CT Pathology Detection API", "endpoints": ["/predict", "/health", "/docs"]}
