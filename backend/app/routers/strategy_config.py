"""Strategy configuration API — read-only endpoint."""

from fastapi import APIRouter

from app.strategy_config_loader import get_strategy_config

router = APIRouter(prefix="/api", tags=["strategy-config"])


@router.get("/strategy-config")
async def get_strategy_config_endpoint() -> dict:
    cfg = get_strategy_config()
    return {
        "params": {
            name: {
                "default": meta.default,
                "min": meta.min,
                "max": meta.max,
                "step": meta.step,
                "integer": meta.integer,
                "label": meta.label,
                "description": meta.description,
            }
            for name, meta in cfg.params.items()
        },
        "presets": cfg.presets,
    }
