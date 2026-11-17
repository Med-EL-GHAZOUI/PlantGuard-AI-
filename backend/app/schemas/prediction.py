from pydantic import BaseModel, ConfigDict

DISCLAIMER = "Outil expérimental d'aide à l'identification, pas un diagnostic professionnel."


class ClassScore(BaseModel):
    class_name: str
    confidence: float


class PredictionResponse(BaseModel):
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
