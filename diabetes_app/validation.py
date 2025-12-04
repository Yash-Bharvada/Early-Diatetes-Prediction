from __future__ import annotations

from pydantic import BaseModel, confloat, conint


class HealthInput(BaseModel):
    Pregnancies: conint(ge=0, le=20) = 0
    Glucose: conint(ge=0, le=300)
    BloodPressure: conint(ge=40, le=200)
    SkinThickness: conint(ge=0, le=100)
    Insulin: conint(ge=0, le=1000)
    BMI: confloat(ge=10.0, le=70.0)
    DiabetesPedigreeFunction: confloat(ge=0.0, le=3.0)
    Age: conint(ge=10, le=120)

    def to_array(self, feature_names: list[str]):
        values = []
        for name in feature_names:
            values.append(getattr(self, name))
        return values
