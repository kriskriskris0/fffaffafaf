# setup.ps1
Write-Host "🚀 Начинаем настройку проекта RAG с Qwen" -ForegroundColor Green

# Удаляем старое окружение, если есть
if (Test-Path ".venv") {
    Write-Host "📦 Удаляем старое виртуальное окружение..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force .venv
}

# Создаём новое виртуальное окружение
Write-Host "🔨 Создаём виртуальное окружение..." -ForegroundColor Yellow
python -m venv .venv

# Активируем
& .venv\Scripts\Activate.ps1

# Обновляем pip
python -m pip install --upgrade pip

# Устанавливаем зависимости из requirements.txt
Write-Host "📦 Устанавливаем зависимости (это займёт несколько минут)..." -ForegroundColor Yellow
pip install -r requirements.txt

# Проверяем, что torch встал корректно
python -c "import torch; print(f'PyTorch {torch.__version__} установлен корректно')"

Write-Host "✅ Проект готов!" -ForegroundColor Green
Write-Host "👉 Для запуска выполните: streamlit run app.py" -ForegroundColor Cyan