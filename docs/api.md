# API Documentation

This project exposes a Streamlit UI for interactive risk prediction. For programmatic access, use the Python package functions.

- `diabetes_app.validation.HealthInput` – Typed input model with bounds and conversion to array.
- `diabetes_app.predictor.get_predictor()` – Returns a callable `predict(payload_array)` that outputs `{ prediction, probability, feature_names }`.

Inputs must be ordered according to `feature_names` returned by the predictor.

## Example
```python
from diabetes_app import get_predictor, HealthInput
predict = get_predictor()
fi = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]
payload = HealthInput(Pregnancies=1, Glucose=120, BloodPressure=80, SkinThickness=20, Insulin=90, BMI=25.0, DiabetesPedigreeFunction=0.5, Age=35).to_array(fi)
print(predict(payload))
```
