from typing import Iterator

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, Response, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from backend.app.schemas.prediction import (
    AnalysisOut, AnalysisPage, DiseaseInfo, ErrorResponse, HealthResponse, PredictionResponse,
    StatsOut,
)
from backend.app.services import analysis_service
from backend.app.services.knowledge_base import describe_classes
from backend.app.services.prediction_service import PredictionService
from backend.app.utils.upload_validation import UploadValidationError, validate_upload

router = APIRouter(prefix="/api")


def get_service(request: Request) -> PredictionService:
    service = request.app.state.prediction_service
    if service is None:
        raise HTTPException(503, "Modèle indisponible : le service n'est pas prêt.")
    return service


def get_db(request: Request) -> Iterator[Session]:
    with request.app.state.session_factory() as session:
        yield session


def _image_url(analysis_id: int) -> str:
    return f"/api/analyses/{analysis_id}/image"


def _to_out(row) -> AnalysisOut:
    fields = {name: getattr(row, name) for name in AnalysisOut.model_fields if name != "image_url"}
    return AnalysisOut(**fields, image_url=_image_url(row.id))


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    loaded = request.app.state.prediction_service is not None
    return HealthResponse(status="ok" if loaded else "degraded", model_loaded=loaded,
                          model_name=request.app.state.model_name)


@router.post("/predict", response_model=PredictionResponse,
             responses={400: {"model": ErrorResponse}, 413: {"model": ErrorResponse},
                        415: {"model": ErrorResponse}, 503: {"model": ErrorResponse}})
def predict(request: Request, file: UploadFile = File(...),
            service: PredictionService = Depends(get_service),
            db: Session = Depends(get_db)) -> PredictionResponse:
    settings = request.app.state.settings
    data = file.file.read(settings.max_upload_bytes + 1)  # lecture bornée : jamais un fichier géant en mémoire
    try:
        image_format = validate_upload(file.filename, file.content_type, data, settings.max_upload_bytes)
    except UploadValidationError as exc:
        raise HTTPException(exc.status_code, exc.message) from None
    prediction = service.predict(data)
    row = analysis_service.save_analysis(db, settings.upload_dir, data, image_format, prediction)
    return PredictionResponse(**prediction.to_dict(), id=row.id, image_url=_image_url(row.id))


@router.get("/analyses", response_model=AnalysisPage)
def list_analyses(limit: int = Query(20, ge=1, le=100), offset: int = Query(0, ge=0),
                  db: Session = Depends(get_db)) -> AnalysisPage:
    total, rows = analysis_service.list_analyses(db, limit, offset)
    return AnalysisPage(total=total, items=[_to_out(r) for r in rows])


@router.get("/analyses/{analysis_id}", response_model=AnalysisOut,
            responses={404: {"model": ErrorResponse}})
def get_analysis(analysis_id: int, db: Session = Depends(get_db)) -> AnalysisOut:
    row = analysis_service.get_analysis(db, analysis_id)
    if row is None:
        raise HTTPException(404, "Analyse introuvable.")
    return _to_out(row)


@router.get("/analyses/{analysis_id}/image", responses={404: {"model": ErrorResponse}})
def get_analysis_image(analysis_id: int, request: Request, db: Session = Depends(get_db)):
    row = analysis_service.get_analysis(db, analysis_id)
    path = request.app.state.settings.upload_dir / row.image_filename if row else None
    if path is None or not path.exists():
        raise HTTPException(404, "Image introuvable.")
    return FileResponse(path)


@router.delete("/analyses/{analysis_id}", status_code=204, responses={404: {"model": ErrorResponse}})
def delete_analysis(analysis_id: int, request: Request, db: Session = Depends(get_db)) -> Response:
    if not analysis_service.delete_analysis(db, request.app.state.settings.upload_dir, analysis_id):
        raise HTTPException(404, "Analyse introuvable.")
    return Response(status_code=204)


@router.get("/stats", response_model=StatsOut)
def stats(db: Session = Depends(get_db)) -> dict:
    return analysis_service.compute_stats(db)


@router.get("/diseases", response_model=list[DiseaseInfo])
def diseases(request: Request) -> list[dict]:
    return describe_classes(request.app.state.class_names, request.app.state.knowledge)
