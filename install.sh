set -e

echo "📦 Установка CT-CLIP зависимостей..."

# transformer_maskgit
cd src/transformer_maskgit
pip install -e .
cd ../..

# CT-CLIP
cd src/ct_clip
pip install -e .
cd ../..

# Основные зависимости
pip install -r requirements.txt

echo "✅ Установка завершена!"
