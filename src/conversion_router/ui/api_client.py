"""HTTP client for the existing FastAPI endpoints used by the UI.

Responses are parsed into the existing Pydantic contracts from
`conversion_router.schemas`; this module adds no prediction or policy logic.
The full-routing path chains `POST /predict` and `POST /decide` (not `/route`,
which returns only the decision) and verifies both halves share one
`request_id`, so the UI can never pair a decision with a different prediction.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import StrEnum

import httpx
from pydantic import ValidationError

from conversion_router.schemas import (
    DecisionResponse,
    ErrorResponse,
    PredictionResponse,
    SessionFeaturesRequest,
)

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
BASE_URL_ENV_VAR = "CQR_API_BASE_URL"
DEFAULT_TIMEOUT_SECONDS = 10.0


class ApiErrorKind(StrEnum):
    UNREACHABLE = "UNREACHABLE"
    TIMEOUT = "TIMEOUT"
    INVALID_INPUT = "INVALID_INPUT"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    CONTRACT_VIOLATION = "CONTRACT_VIOLATION"
    HTTP_ERROR = "HTTP_ERROR"


class ApiError(RuntimeError):
    def __init__(self, kind: ApiErrorKind, message: str, request_id: str | None = None):
        super().__init__(message)
        self.kind = kind
        self.message = message
        self.request_id = request_id


@dataclass(frozen=True)
class ReadyInfo:
    model_name: str
    model_version: str
    feature_contract_version: str


@dataclass(frozen=True)
class RoutedResult:
    """One prediction and the decision made from exactly that prediction."""

    prediction: PredictionResponse
    decision: DecisionResponse

    @property
    def request_id(self) -> str:
        return self.prediction.request_id


class ApiClient:
    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = http_client or httpx.Client(base_url=self.base_url, timeout=timeout)

    @classmethod
    def from_env(cls) -> ApiClient:
        return cls(base_url=os.environ.get(BASE_URL_ENV_VAR, DEFAULT_BASE_URL))

    # -- transport -------------------------------------------------------

    def _request(self, method: str, path: str, json_body: dict | None = None) -> httpx.Response:
        try:
            return self._client.request(method, path, json=json_body)
        except httpx.TimeoutException as exc:
            raise ApiError(
                ApiErrorKind.TIMEOUT, "The decision service did not respond in time."
            ) from exc
        except httpx.TransportError as exc:
            raise ApiError(
                ApiErrorKind.UNREACHABLE,
                f"The decision service at {self.base_url} is not reachable.",
            ) from exc

    @staticmethod
    def _raise_for_error(response: httpx.Response) -> None:
        if response.status_code < 400:
            return
        request_id = None
        message = f"Unexpected HTTP {response.status_code} from the decision service."
        try:
            error = ErrorResponse.model_validate(response.json())
            request_id = error.request_id
            message = error.error.message
        except (ValueError, ValidationError):
            pass
        if response.status_code == 422:
            raise ApiError(ApiErrorKind.INVALID_INPUT, "The service rejected the session input.")
        if response.status_code == 503:
            raise ApiError(ApiErrorKind.MODEL_UNAVAILABLE, message, request_id)
        raise ApiError(ApiErrorKind.HTTP_ERROR, message, request_id)

    @staticmethod
    def _parse(model, response: httpx.Response):
        try:
            return model.model_validate(response.json())
        except (ValueError, ValidationError) as exc:
            raise ApiError(
                ApiErrorKind.CONTRACT_VIOLATION,
                f"The service returned a response that does not match {model.__name__}.",
            ) from exc

    # -- endpoints -------------------------------------------------------

    def health(self) -> bool:
        response = self._request("GET", "/health")
        return response.status_code == 200

    def ready(self) -> ReadyInfo:
        response = self._request("GET", "/ready")
        self._raise_for_error(response)
        body = response.json()
        try:
            return ReadyInfo(
                model_name=body["model_name"],
                model_version=body["model_version"],
                feature_contract_version=body["feature_contract_version"],
            )
        except (KeyError, TypeError) as exc:
            raise ApiError(
                ApiErrorKind.CONTRACT_VIOLATION, "Unexpected /ready response shape."
            ) from exc

    def predict(self, session: SessionFeaturesRequest) -> PredictionResponse:
        response = self._request("POST", "/predict", session.model_dump(mode="json"))
        self._raise_for_error(response)
        return self._parse(PredictionResponse, response)

    def decide(self, prediction: PredictionResponse) -> DecisionResponse:
        response = self._request("POST", "/decide", prediction.model_dump(mode="json"))
        self._raise_for_error(response)
        return self._parse(DecisionResponse, response)

    def route(self, session: SessionFeaturesRequest) -> RoutedResult:
        """Model prediction followed by the decision for that same prediction."""
        prediction = self.predict(session)
        return self.pair(prediction, self.decide(prediction))

    @staticmethod
    def pair(prediction: PredictionResponse, decision: DecisionResponse) -> RoutedResult:
        """Bind a decision to its prediction, refusing a mismatched request ID."""
        if decision.request_id != prediction.request_id:
            raise ApiError(
                ApiErrorKind.CONTRACT_VIOLATION,
                "Decision request_id does not match the prediction it was made from.",
                prediction.request_id,
            )
        return RoutedResult(prediction=prediction, decision=decision)
