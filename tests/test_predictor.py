from diabetes_app import get_predictor


def test_predictor_runs():
    predict = get_predictor()
    # Use reasonable sample input order from feature_names
    result_meta = predict([0] * 8)
    feature_names = result_meta["feature_names"]
    assert isinstance(feature_names, list)
    # Create plausible values
    sample = {
        "Pregnancies": 1,
        "Glucose": 120,
        "BloodPressure": 80,
        "SkinThickness": 20,
        "Insulin": 90,
        "BMI": 25.0,
        "DiabetesPedigreeFunction": 0.5,
        "Age": 35,
    }
    ordered = [sample[n] for n in feature_names]
    result = predict(ordered)
    assert "prediction" in result and "probability" in result

