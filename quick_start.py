#!/usr/bin/env python3
"""Quick Start скрипт для CT Pathology Detection"""

from pathlib import Path
from src.pipeline.core_pipeline import CTPathologyPipeline
from src.pipeline.data_models import PipelineConfig

def main():
    print("🏥 CT Pathology Detection - Quick Start\n")
    
    # Конфигурация
    config = PipelineConfig(
        ct_clip_checkpoint="models/CT_LiPro_v2.pt",
        catboost_model="models/catboost_pathology_classifier.cbm",
        device="cuda"
    )
    
    # Инициализация
    print("⏳ Инициализация pipeline...")
    pipeline = CTPathologyPipeline(config)
    print("✅ Pipeline готов\n")
    
    # Обработка
    zip_paths = ["data/study1.zip"]  # Укажите свои файлы
    
    results = pipeline.process_zip_archives(
        zip_paths=zip_paths,
        output_excel="results.xlsx"
    )
    
    print(f"\n✅ Обработано: {len(results)} исследований")
    print("📁 Результаты: results.xlsx")

if __name__ == "__main__":
    main()
