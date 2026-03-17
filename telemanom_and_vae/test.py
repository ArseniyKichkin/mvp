import dotenv
import numpy as np
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field
import mlflow
import logging
from typing import List, Optional, Any
import uvicorn
from prometheus_client import Gauge, generate_latest

prediction_metric = Gauge(
    "anomaly_score_value",
    "Last anomaly score returned by the model",
["model_name", "model_version"]
)



dotenv.load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Drift Model Inference Service",
    description="Сервис для инференса модели обнаружения дрейфа",
    version="1.0.0"
)


# Конфигурация модели
MODEL_NAME = "drift_model"
MODEL_VERSION = "latest"  # или конкретная версия, например "1"

# Глобальная переменная для модели
model = None


# Pydantic модели для запросов и ответов
class PredictionRequest(BaseModel):
    data: List[List[List[float]]] = Field(
        ...,
        description="Входные данные для модели. Ожидается список списков формы (batch, 55) или (batch, 250, 55)"
    )
    batch_size: Optional[int] = Field(100, description="Размер батча для инференса")


class PredictionResponse(BaseModel):
    predictions: List[Any]
    shape: List[int]
    model_info: dict


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_uri: str
    service: str


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")

# Загрузка модели при старте приложения
@app.on_event("startup")
async def load_model():
    """Загружает модель MLflow при старте приложения"""
    global model
    try:
        model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
        logger.info(f"Загрузка модели из {model_uri}")

        model = mlflow.pyfunc.load_model(model_uri)

        logger.info(f"Модель успешно загружена")
        logger.info(f"Тип модели: {type(model)}")

        # Пробуем получить метаданные модели
        if hasattr(model, 'metadata'):
            logger.info(f"Метаданные модели\n: {model.metadata}")

    except Exception as e:
        logger.error(f"Ошибка загрузки модели: {e}")
        raise e


@app.get("/", response_model=HealthResponse)
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Проверка здоровья сервиса"""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        model_uri=f"models:/{MODEL_NAME}/{MODEL_VERSION}",
        service="drift-model-inference"
    )


@app.get("/info")
async def model_info():
    """Информация о модели"""
    if model is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")

    info = {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "model_type": str(type(model)),
    }

    # Добавляем метаданные если есть
    if hasattr(model, 'metadata'):
        if hasattr(model.metadata, 'signature'):
            info["input_schema"] = str(model.metadata.signature.inputs)
            info["output_schema"] = str(model.metadata.signature.outputs)

    return info


@app.post("/invocations", response_model=PredictionResponse)
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Эндпоинт для инференса модели

    Ожидает данные в формате:
    {
        "data": [[0.1, 0.2, ...], [0.3, 0.4, ...]],
        "batch_size": 100
    }

    Где внутренние списки должны иметь длину 55 (признаки)
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")

    try:
        # Преобразуем входные данные в numpy array
        input_data = np.array(request.data)
        logger.info(f"Получены данные формы: {input_data.shape}")

        # Проверяем размерность
        if input_data.ndim == 2:
            # Если пришли данные (batch, features), используем как есть
            logger.info(f"2D вход: {input_data.shape}")
        elif input_data.ndim == 3:
            # Если пришли (batch, timesteps, features), используем как есть
            logger.info(f"3D вход: {input_data.shape}")
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Ожидается 2D или 3D массив, получен {input_data.ndim}D"
            )

        # Делаем предсказания
        predictions = model.predict(input_data)
        logger.info(f"Предсказания получены. Форма: {predictions.shape if hasattr(predictions, 'shape') else 'scalar'}")

        if len(predictions) != 0:
            pred_value = predictions[-1]

            prediction_metric.labels(
                model_name=MODEL_NAME,
                model_version=MODEL_VERSION
            ).set(pred_value)


        # Преобразуем в список для JSON ответа
        if hasattr(predictions, 'tolist'):
            pred_list = predictions.tolist()
        else:
            pred_list = list(predictions)

        return PredictionResponse(
            predictions=pred_list,
            shape=list(predictions.shape) if hasattr(predictions, 'shape') else [],
            model_info={
                "model_name": MODEL_NAME,
                "model_version": MODEL_VERSION
            }
        )

    except Exception as e:
        logger.error(f"Ошибка при предсказании: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/invocations/batch")
async def predict_batch(requests: List[PredictionRequest]):
    """
    Пакетная обработка нескольких запросов
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")

    results = []
    for i, req in enumerate(requests):
        try:
            input_data = np.array(req.data)
            predictions = model.predict(input_data)

            if hasattr(predictions, 'tolist'):
                pred_list = predictions.tolist()
            else:
                pred_list = list(predictions)

            results.append({
                "request_id": i,
                "success": True,
                "predictions": pred_list,
                "shape": list(predictions.shape) if hasattr(predictions, 'shape') else []
            })
        except Exception as e:
            results.append({
                "request_id": i,
                "success": False,
                "error": str(e)
            })

    return {"results": results}