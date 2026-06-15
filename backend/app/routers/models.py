"""Model configuration API — list selectable preset models.

Returns safe preset metadata only; api_keys are never exposed.
"""

from fastapi import APIRouter

from app.llm.model_resolver import list_presets

router = APIRouter(prefix="/api", tags=["models"])


@router.get("/models")
async def get_models_endpoint() -> dict:
    return {"presets": list_presets()}
