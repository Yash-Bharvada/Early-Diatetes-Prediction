translations = {
    "en": {
        "title": "Diabetes Risk Predictor",
        "subtitle": "Interactive health assessment tool for early diabetes detection",
        "tab_input": "Input Data",
        "tab_results": "Results",
        "calc_button": "Calculate Risk Score",
        "glucose_insulin": "Glucose & Insulin",
        "cardio": "Cardiovascular",
        "body_metrics": "Body Metrics",
        "demographics": "Demographics",
        "glucose": "Glucose Level (mg/dL)",
        "insulin": "Insulin (μU/mL)",
        "bp": "Blood Pressure (mmHg)",
        "skin": "Skin Thickness (mm)",
        "bmi": "BMI (Body Mass Index)",
        "dpf": "Diabetes Pedigree",
        "age": "Age (years)",
        "preg": "Number of Pregnancies",
        "results_title": "Risk Assessment Results",
        "results_sub": "Based on your provided health metrics",
        "low_risk": "Low Risk",
        "high_risk": "High Risk",
        "breakdown": "Risk Factor Breakdown",
        "breakdown_sub": "Individual health metrics analysis",
        "recommendations": "Recommendations",
    },
    "hi": {
        "title": "मधुमेह जोखिम भविष्यवक्ता",
        "subtitle": "प्रारंभिक मधुमेह पहचान के लिए इंटरैक्टिव स्वास्थ्य आकलन",
        "tab_input": "इनपुट डेटा",
        "tab_results": "परिणाम",
        "calc_button": "जोखिम स्कोर की गणना करें",
        "glucose_insulin": "ग्लूकोज़ और इंसुलिन",
        "cardio": "हृदय संबंधी",
        "body_metrics": "शरीर के माप",
        "demographics": "जनसांख्यिकी",
        "glucose": "ग्लूकोज़ स्तर (mg/dL)",
        "insulin": "इंसुलिन (μU/mL)",
        "bp": "रक्तचाप (mmHg)",
        "skin": "त्वचा मोटाई (mm)",
        "bmi": "बीएमआई",
        "dpf": "मधुमेह वंशावली",
        "age": "आयु (वर्ष)",
        "preg": "गर्भधारण की संख्या",
        "results_title": "जोखिम मूल्यांकन परिणाम",
        "results_sub": "आपके स्वास्थ्य मेट्रिक्स के आधार पर",
        "low_risk": "कम जोखिम",
        "high_risk": "उच्च जोखिम",
        "breakdown": "जोखिम कारक विभाजन",
        "breakdown_sub": "व्यक्तिगत स्वास्थ्य मेट्रिक्स विश्लेषण",
        "recommendations": "सिफारिशें",
    },
}


def get_translator(lang: str):
    data = translations.get(lang, translations["en"])
    def t(key: str):
        return data.get(key, key)
    return t

