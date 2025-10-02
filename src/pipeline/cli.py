# src/pipeline/cli.py
"""
Command Line Interface для тестирования Core Pipeline
"""
import argparse
import sys
from pathlib import Path
import json

from .core_pipeline import CTPathologyPipeline
from .data_models import PipelineConfig

def main():
    parser = argparse.ArgumentParser(description="CT Pathology Detection Pipeline")
    parser.add_argument("zip_paths", nargs="+", help="Пути к ZIP архивам")
    parser.add_argument("--ctclip-checkpoint", required=True, help="Путь к CT-CLIP checkpoint")
    parser.add_argument("--catboost-model", required=True, help="Путь к CatBoost модели")
    parser.add_argument("--output", "-o", help="Выходной Excel файл")
    parser.add_argument("--max-workers", type=int, default=4, help="Количество воркеров")
    parser.add_argument("--device", default="auto", help="Device для моделей")
    parser.add_argument("--log-level", default="INFO", help="Уровень логирования")
    
    args = parser.parse_args()
    
    # Проверяем входные файлы
    for zip_path in args.zip_paths:
        if not Path(zip_path).exists():
            print(f"❌ ZIP файл не найден: {zip_path}")
            sys.exit(1)
    
    if not Path(args.ctclip_checkpoint).exists():
        print(f"❌ CT-CLIP checkpoint не найден: {args.ctclip_checkpoint}")
        sys.exit(1)
        
    if not Path(args.catboost_model).exists():
        print(f"❌ CatBoost модель не найдена: {args.catboost_model}")
        sys.exit(1)
    
    # Конфигурация pipeline
    config = PipelineConfig(
        ctclip_checkpoint=args.ctclip_checkpoint,
        catboost_model=args.catboost_model,
        device=args.device,
        max_workers=args.max_workers,
        log_level=args.log_level
    )
    
    # Запуск pipeline
    try:
        pipeline = CTPathologyPipeline(config)
        excel_path, statistics = pipeline.process_zip_archives(
            args.zip_paths, 
            args.output
        )
        
        print(f"✅ Обработка завершена!")
        print(f"📊 Excel отчёт: {excel_path}")
        print(f"📈 Статистика:")
        for key, value in statistics.items():
            print(f"   {key}: {value}")
            
    except Exception as e:
        print(f"❌ Ошибка pipeline: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
