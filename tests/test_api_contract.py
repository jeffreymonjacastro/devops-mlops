import pytest
from pydantic import ValidationError

from src.api.schemas import HousePredictionRequest


def test_prediction_requires_square_footage():
    payload = {
        "bedrooms": 2,
        "bathrooms": 1.0,
        "location": "Rural",
        "year_built": 1948,
        "condition": "Poor",
    }

    with pytest.raises(ValidationError) as error:
        HousePredictionRequest.model_validate(payload)

    violations = error.value.errors()
    assert len(violations) == 1
    assert violations[0]["loc"] == ("sqft",)
    assert violations[0]["type"] == "missing"
