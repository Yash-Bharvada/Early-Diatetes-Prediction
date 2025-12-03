from diabetes_app.validation import HealthInput


def test_validation_bounds():
    inp = HealthInput(
        Pregnancies=1,
        Glucose=120,
        BloodPressure=80,
        SkinThickness=20,
        Insulin=90,
        BMI=25.0,
        DiabetesPedigreeFunction=0.5,
        Age=35,
    )
    arr = inp.to_array([
        "Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"
    ])
    assert len(arr) == 8

