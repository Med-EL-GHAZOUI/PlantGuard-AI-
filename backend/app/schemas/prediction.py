from datetime import datetime

from pydantic import BaseModel, ConfigDict

DISCLAIMER = "Outil expérimental d'aide à l'identification, pas un diagnostic professionnel."


class ClassScore(BaseModel):
    class_name: str
    confidence: float


class PredictionResponse(BaseModel):
    id: int
    image_url: str
    plant: str
    disease: str
    class_name: str
    confidence: float
    is_healthy: bool
    low_confidence: bool
    top_k: list[ClassScore]
    disclaimer: str = DISCLAIMER


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loaded: bool
    model_name: str | None = None


class DiseaseInfo(BaseModel):
    class_name: str
    plant: str
    disease: str
    is_healthy: bool
    description: str | None = None
    advice: str | None = None


class ErrorResponse(BaseModel):
    detail: str


class AnalysisOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: datetime
    class_name: str
    plant: str
    disease: str
    confidence: float
    is_healthy: bool
    low_confidence: bool
    image_url: str


class AnalysisPage(BaseModel):
    total: int
    items: list[AnalysisOut]


class DiseaseCount(BaseModel):
    class_name: str
    count: int


class PlantCount(BaseModel):
    plant: str
    count: int


class DayCount(BaseModel):
    date: str
    count: int


class StatsOut(BaseModel):
    total_analyses: int
    healthy_count: int
    diseased_count: int
    by_disease: list[DiseaseCount]
    top_plants: list[PlantCount]
    per_day: list[DayCount]
