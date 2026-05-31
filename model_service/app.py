import json
from pathlib import Path
from datetime import datetime
from typing import List
import threading
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from lightgbm import Booster
from contextlib import asynccontextmanager

# Определяем пути к артефактам
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "model.txt"
FEATURE_COLS_PATH = MODELS_DIR / "feature_cols.json"
METRICS_PATH = MODELS_DIR / "metrics.json"

# Глобальное состояние сервиса (In-Memory Реестр моделей)
CURRENT_MODEL = None
FEATURE_COLS = None
MODEL_METRICS = {}
MODEL_VERSION = "unknown"

# Блокировка для потокобезопасной замены модели
model_lock = threading.RLock()


def reload_model_from_disk():
    """
    Потокобезопасная функция для перезагрузки модели с диска в память.
    Позволяет обновлять модель на лету без перезапуска всего API.
    """
    global CURRENT_MODEL, FEATURE_COLS, MODEL_METRICS, MODEL_VERSION

    # гарантируем, что замена модели не сломает инференс
    with model_lock:
        print("Запрос на обновление модели в памяти")

        if MODEL_PATH.exists() and FEATURE_COLS_PATH.exists():
            try:
                # Загружаем бинарник LightGBM
                CURRENT_MODEL = Booster(model_file=str(MODEL_PATH))

                # Загружаем фичи
                with open(FEATURE_COLS_PATH, 'r') as f:
                    FEATURE_COLS = json.load(f)

                # Загружаем метрики качества
                if METRICS_PATH.exists():
                    with open(METRICS_PATH, 'r') as f:
                        MODEL_METRICS = json.load(f)

                # Генерируем версию на основе времени изменения файла
                file_time = datetime.fromtimestamp(MODEL_PATH.stat().st_mtime)
                MODEL_VERSION = f"lgbm_v_{file_time.strftime('%Y%m%d_%H%M%S')}"

                print(f"Успех: Модель {MODEL_VERSION} успешно развернута в памяти!")
                print(f"Текущий MAPE модели на валидации: {MODEL_METRICS.get('cv_mape_mean', 0):.2%}")
                return True
            except Exception as e:
                print(f"Критическая ошибка при чтении файлов модели: {e}")
                return False
        else:
            print("Предупреждение: Файлы модели не найдены. Сервис работает в демо-режиме заглушки.")
            return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Код выполняется при старте приложения
    print("Запуск сервера Demand Forecast API")
    reload_model_from_disk()
    yield
    # Код выполняется при остановке приложения
    print("Остановка сервера")


app = FastAPI(
    title="Demand Forecast API",
    description="Сервис прогнозирования спроса с поддержкой динамической смены моделей (Blue-Green)",
    version="1.1.0",
    lifespan=lifespan
)


# --- Схемы данных (Pydantic) ---

class PredictionRequest(BaseModel):
    product_id: int = Field(..., description="ID товара", example=1)
    day_of_week: int = Field(..., ge=0, le=6, description="День недели (0-6)")
    day_of_month: int = Field(..., ge=1, le=31)
    month: int = Field(..., ge=1, le=12)
    is_weekend: int = Field(..., ge=0, le=1)
    price: float = Field(..., gt=0, description="Текущая цена товара")
    inventory: int = Field(..., ge=0, description="Остаток на складе")
    lag_1: float = Field(..., description="Продажи вчера")
    lag_7: float = Field(..., description="Продажи неделю назад")
    lag_14: float = Field(..., description="Продажи 2 недели назад")
    rolling_mean_7: float = Field(..., description="Скользящее среднее за 7 дней")


class PredictionResponse(BaseModel):
    product_id: int
    predicted_sales: int
    model_version: str
    timestamp: str


class BatchPredictionRequest(BaseModel):
    requests: List[PredictionRequest]


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]


def prepare_features(req: PredictionRequest) -> np.ndarray:
    """Преобразует Pydantic-запрос в массив для LightGBM"""
    features = [
        req.day_of_week, req.day_of_month, req.month, req.is_weekend,
        req.price, req.inventory, req.lag_1, req.lag_7, req.lag_14, req.rolling_mean_7
    ]
    return np.array(features).reshape(1, -1)


# --- Эндпоинты ---

@app.get("/")
async def root():
    return {
        "service": "Demand Forecast API",
        "status": "online",
        "active_model_version": MODEL_VERSION,
        "is_model_loaded": CURRENT_MODEL is not None
    }


@app.get("/health")
async def health_check():
    if CURRENT_MODEL is None:
        return {
            "status": "degraded",
            "reason": "Модель не загружена. Сервис не может делать предсказания.",
            "timestamp": datetime.now().isoformat()
        }

    return {
        "status": "healthy",
        "model_version": MODEL_VERSION,
        "metrics_on_cv": {
            "mape": MODEL_METRICS.get("cv_mape_mean"),
            "samples_trained": MODEL_METRICS.get("n_samples")
        },
        "timestamp": datetime.now().isoformat()
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if CURRENT_MODEL is None:
        raise HTTPException(status_code=503, detail="Модель временно недоступна")

    try:
        features = prepare_features(request)
        raw_prediction = CURRENT_MODEL.predict(features)[0]
        final_prediction = max(0, int(np.ceil(raw_prediction)))

        return PredictionResponse(
            product_id=request.product_id,
            predicted_sales=final_prediction,
            model_version=MODEL_VERSION,
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка инференса: {str(e)}")


@app.post("/batch_predict", response_model=BatchPredictionResponse)
async def batch_predict(request: BatchPredictionRequest):
    if CURRENT_MODEL is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")

    try:
        feature_list = [prepare_features(req)[0] for req in request.requests]
        X_matrix = np.array(feature_list)
        raw_predictions = CURRENT_MODEL.predict(X_matrix)

        responses = []
        for req, raw_pred in zip(request.requests, raw_predictions):
            responses.append(PredictionResponse(
                product_id=req.product_id,
                predicted_sales=max(0, int(np.ceil(raw_pred))),
                model_version=MODEL_VERSION,
                timestamp=datetime.now().isoformat()
            ))
        return BatchPredictionResponse(predictions=responses)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка пакетного расчета: {str(e)}")


@app.post("/model/reload")
async def trigger_model_reload(background_tasks: BackgroundTasks):
    """
    Эндпоинт для MLOps пайплайна.
    Вызывается после успешного обучения новой модели.
    Трафик переключается на новую модель без остановки сервера.
    """
    background_tasks.add_task(reload_model_from_disk)
    return {
        "status": "reload_triggered",
        "message": "Запрошено обновление модели. Процесс запущен в фоне."
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)