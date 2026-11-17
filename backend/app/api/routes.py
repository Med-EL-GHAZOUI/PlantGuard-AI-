from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile

from backend.app.schemas.prediction import (
    DiseaseInfo, ErrorResponse, HealthResponse, PredictionResponse,
)
from backend.app.services.knowledge_base import describe_classes
from backend.app.services.prediction_service import PredictionService
from backend.app.utils.upload_validation import UploadValidationError, validate_upload

router = APIRouter(prefix="/api")


def get_service(request: Request) -> PredictionService:
    service = request.app.state.prediction_service
    if service is None:
        raise HTTPException(503, "Modèle indisponible : le service n'est pas prêt.")
    return service


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    loaded = request.app.state.prediction_service is not None
    return HealthResponse(status="ok" if loaded else "degraded", model_loaded=loaded,
                          model_name=request.app.state.model_name)


@router.post("/predict", response_model=PredictionResponse,
             responses={400: {"model": ErrorResponse}, 413: {"model": ErrorResponse},
                        415: {"model": ErrorResponse}, 503: {"model": ErrorResponse}})
def predict(request: Request, file: UploadFile = File(...),
            service: PredictionService = Depends(get_service)) -> PredictionResponse:
    max_bytes = request.app.state.settings.max_upload_bytes
    data = file.file.read(max_bytes + 1)  # lit au plus max+1 octets : jamais un fichier géant en mémoire
    try:
        validate_upload(file.filename, file.content_type, data, max_bytes)
    except UploadValidationError as exc:
        raise HTTPException(exc.status_code, exc.message) from None
    prediction = service.predict(data)
    return PredictionResponse(**prediction.to_dict())


@router.get("/diseases", response_model=list[DiseaseInfo])
def diseases(request: Request) -> list[dict]:
    return describe_classes(request.app.state.class_names, request.app.state.knowledge)
