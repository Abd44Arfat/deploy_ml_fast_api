import joblib
import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

# Load vectorizer and model
try:
    vectorizer = joblib.load("vectorizer.jbl")
    with open("model1.pkl", "rb") as f:
        model = pickle.load(f)
except Exception as e:
    raise RuntimeError(f"Error loading model or vectorizer: {e}")

class TextRequest(BaseModel):
    text: str

@app.post("/predict")
def predict_language(request: TextRequest):
    try:
        # Ensure correct input format
        transformed_text = vectorizer.transform([request.text])
        
        # Check if model is of the correct type
        if not hasattr(model, 'predict'):
            raise RuntimeError("Loaded model does not have a 'predict' method.")
        
        prediction = model.predict(transformed_text)[0]  # Perform prediction
        return {"prediction": (prediction)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))