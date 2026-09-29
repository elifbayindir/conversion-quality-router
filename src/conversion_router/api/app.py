"""FastAPI application factory: health, readiness, and prediction only.

Loads frozen model artifacts exactly once at startup (not per-request) and
stores the result (or the error that prevented loading) on `app.state`, so
`/health` never depends on it and `/ready` and `/predict` fail safely
(503, not a crash) when artifacts are missing or fail integrity checks.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from conversion_router.modeling.inference import (
    ArtifactIntegrityError,
    ArtifactUnavailableError,
    LoadedArtifacts,
    load_artifacts,
)

logger = logging.getLogger("conversion_router.api")


class ArtifactState:
    """Holds the once-loaded artifacts, or the error that prevented loading."""

    def __init__(self) -> None:
        self.artifacts: LoadedArtifacts | None = None
        self.load_error: Exception | None = None

    def try_load(self) -> None:
        try:
            self.artifacts = load_artifacts()
            self.load_error = None
        except (ArtifactUnavailableError, ArtifactIntegrityError) as exc:
            self.artifacts = None
            self.load_error = exc
            logger.warning("Artifact loading failed: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.artifact_state = ArtifactState()
    app.state.artifact_state.try_load()
    # Unset in production: get_decision() builds a real AnthropicProvider lazily.
    # Tests set this to a fake/mock provider so no run ever calls the network.
    app.state.llm_provider = None
    yield


def create_app() -> FastAPI:
    from conversion_router.api.routes import router

    app = FastAPI(title="Conversion Quality Router API", version="1.0.0", lifespan=lifespan)
    app.include_router(router)
    return app


app = create_app()
