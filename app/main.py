from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os
import io

app = FastAPI(title="AI Learning Intelligence Tool")

# --- Load Artifacts ---
MODEL_PATH = os.path.join("models", "completion_model.pkl")
SCALER_PATH = os.path.join("models", "scaler.pkl")
DIFFICULTY_PATH = os.path.join("models", "difficulty_stats.csv")

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    difficulty_stats = pd.read_csv(DIFFICULTY_PATH)
    print("Models loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")
    model = None
    scaler = None
    difficulty_stats = None

# --- Data Models ---
class StudentData(BaseModel):
    student_id: int
    course_id: str
    chapter_id: int
    time_spent: float
    score: float

# --- Helper Functions ---
def preprocess_student_data(df):
    """Aggregates raw student data for the model."""
    # This logic must match training preprocessing
    student_features = df.groupby('student_id').agg({
        'score': ['mean', 'min', 'max', 'std'],
        'time_spent': ['sum', 'mean', 'std'],
        'chapter_id': 'max'
    })
    
    student_features.columns = ['_'.join(col).strip() for col in student_features.columns.values]
    student_features.fillna(0, inplace=True)
    return student_features

# --- Endpoints ---

@app.post("/predict")
async def predict_completion(file: UploadFile = File(...)):
    """
    Predicts completion for a batch of students from an uploaded CSV.
    Input: CSV with columns [student_id, course_id, chapter_id, time_spent, score]
    """
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded.")
    
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Preprocess
        features = preprocess_student_data(df)
        
        # Scale
        X_scaled = scaler.transform(features)
        
        # Predict
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)[:, 1]
        
        # Results
        results = []
        for idx, (student_id, row) in enumerate(features.iterrows()):
            prob = probabilities[idx]
            pred = int(predictions[idx])
            
            # Risk Logic
            risk_level = "Low"
            if prob < 0.3:
                risk_level = "High"
            elif prob < 0.7:
                risk_level = "Medium"
            
            results.append({
                "student_id": int(student_id),
                "completion_probability": round(float(prob), 2),
                "predicted_completion": bool(pred),
                "risk_level": risk_level
            })
            
        return JSONResponse(content={"predictions": results})
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")

@app.get("/difficulty")
def get_difficulty_insights():
    """Returns chapter difficulty insights."""
    if difficulty_stats is None:
         raise HTTPException(status_code=500, detail="Difficulty stats not loaded.")
    
    data = difficulty_stats.to_dict(orient="records")
    return JSONResponse(content={"difficulty_analysis": data})

@app.get("/insights")
def get_general_insights():
    """Returns general insights about the model/training data."""
    # In a real app, this might analyze the live database.
    # Here we return static info or info from the loaded stats.
    return JSONResponse(content={
        "message": "AI Learning Intelligence Tool is active.",
        "model_type": "Random Forest Classifier",
        "features_used": ["avg_score", "total_time", "progress"],
        "version": "1.0.0"
    })

# Serve Static Files (Frontend)
app.mount("/", StaticFiles(directory="app/static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
