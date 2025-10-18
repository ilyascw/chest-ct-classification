from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from fastapi.responses import FileResponse
import shutil
from pathlib import Path
import uuid
import sys
import os
import torch

# Добавляем корень проекта в sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.pipeline.core_pipeline import CTPathologyPipeline
from src.pipeline.data_models import PipelineConfig

app = FastAPI(
    title="CT Pathology Detection API",
    description="API для обнаружения патологий в КТ снимках грудной клетки",
    version="1.0.0"
)

pipeline = None


def cleanup_temp_dir(temp_dir: Path):
    """
    Фоновая задача для удаления временной директории.
    
    Args:
        temp_dir: Путь к временной директории
    """
    try:
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
            print(f"🗑️  Cleaned up: {temp_dir}")
    except Exception as e:
        print(f"⚠️  Failed to cleanup {temp_dir}: {e}")


@app.on_event("startup")
async def startup():
    """Инициализация pipeline при старте приложения"""
    global pipeline
    
    # Автоопределение пути к моделям (Docker или локальный запуск)
    docker_models_path = Path("/app/models")
    local_models_path = Path("models")
    
    if docker_models_path.exists():
        base_path = docker_models_path
        print(f"✅ Using Docker models path: {base_path}")
    elif local_models_path.exists():
        base_path = local_models_path
        print(f"✅ Using local models path: {base_path}")
    else:
        raise FileNotFoundError(
            f"Models directory not found. Tried:\n"
            f"  - {docker_models_path}\n"
            f"  - {local_models_path}\n"
            f"Please ensure models are downloaded."
        )
    
    # Проверка наличия файлов моделей
    ct_clip_path = base_path / "CT_LiPro_v2.pt"
    catboost_path = base_path / "catboost_pathology_classifier.cbm"
    
    if not ct_clip_path.exists():
        raise FileNotFoundError(f"CT-CLIP model not found: {ct_clip_path}")
    if not catboost_path.exists():
        raise FileNotFoundError(f"CatBoost model not found: {catboost_path}")
    
    # Автоопределение device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔧 Using device: {device}")
    
    config = PipelineConfig(
        ct_clip_checkpoint=str(ct_clip_path),
        catboost_model=str(catboost_path),
        device=device,
        max_workers=1, # для стабильности
        log_level="INFO"
    )
    
    pipeline = CTPathologyPipeline(config)
    print("✅ Pipeline initialized successfully")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "status": "running",
        "service": "CT Pathology Detection API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "info": "/info",
            "predict": "/predict (POST with ZIP file)",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health():
    """Проверка состояния сервиса"""
    return {
        "status": "healthy",
        "pipeline_ready": pipeline is not None
    }


@app.post("/predict")
async def predict(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Обработка ZIP архива с DICOM исследованиями.
    
    Args:
        background_tasks: FastAPI background tasks для cleanup
        file: ZIP архив с DICOM файлами
    
    Returns:
        FileResponse: Excel файл с результатами или JSON с ошибкой
    """
    if not file.filename.endswith('.zip'):
        return {"error": "Only ZIP files are supported", "filename": file.filename}
    
    # Создание временной директории для обработки
    job_id = str(uuid.uuid4())[:8]
    temp_dir = Path(f"/tmp/ct_api/{job_id}")
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    zip_path = None
    output_excel = None
    
    try:
        # Сохранение загруженного ZIP
        zip_path = temp_dir / file.filename
        with open(zip_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"📦 Processing ZIP: {zip_path} ({zip_path.stat().st_size / (1024*1024):.1f} MB)")
        
        # Обработка через pipeline
        output_excel = temp_dir / f"results_{job_id}.xlsx"
        
        # Синхронный вызов pipeline (блокирующий)
        result_df = pipeline.process_zip_archives(
            zip_paths=[str(zip_path)],
            output_excel=str(output_excel)
        )
        
        # Проверка существования результата
        if not output_excel.exists():
            return {
                "error": "Pipeline completed but output file was not created",
                "job_id": job_id,
                "expected_path": str(output_excel)
            }
        
        print(f"✅ Results saved: {output_excel} ({output_excel.stat().st_size / 1024:.1f} KB)")
        
        # Планируем очистку после отправки файла
        background_tasks.add_task(cleanup_temp_dir, temp_dir)
        
        # Возврат результатов
        response = FileResponse(
            path=str(output_excel),
            filename=f"ct_pathology_results_{job_id}.xlsx",
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
        return response
    
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"❌ Error processing {file.filename}: {e}")
        print(error_trace)
        
        # Очистка при ошибке
        background_tasks.add_task(cleanup_temp_dir, temp_dir)
        
        return {
            "error": str(e),
            "job_id": job_id,
            "traceback": error_trace,
            "zip_path": str(zip_path) if zip_path else None,
            "output_path": str(output_excel) if output_excel else None
        }


@app.get("/info")
async def info():
    """Информация о конфигурации pipeline"""
    if pipeline is None:
        return {"error": "Pipeline not initialized"}
    
    return {
        "ct_clip_checkpoint": pipeline.config.ct_clip_checkpoint,
        "catboost_model": pipeline.config.catboost_model,
        "device": pipeline.config.device,
        "max_workers": pipeline.config.max_workers,
        "text_prompt": pipeline.config.text_prompt
    }
