"""Local decision-support UI: a thin HTTP client of the FastAPI service.

Nothing in this package loads model artifacts, calibrates, thresholds, or
decides a route. Every number and decision shown comes from the API's
`PredictionResponse` / `DecisionResponse` contracts.
"""
