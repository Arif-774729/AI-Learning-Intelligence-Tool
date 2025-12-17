import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def train_models():
    """Trains and saves the AI models."""
    
    # Load Data
    data_path = os.path.join("data", "student_data.csv")
    if not os.path.exists(data_path):
        print("Data file not found. Run generate_data.py first.")
        return

    df = pd.read_csv(data_path)
    
    # --- Feature Engineering ---
    # Aggregate data to student level for Completion Prediction
    student_features = df.groupby('student_id').agg({
        'score': ['mean', 'min', 'max', 'std'],
        'time_spent': ['sum', 'mean', 'std'],
        'chapter_id': 'max', # Last chapter reached (proxy for progress if we had incomplete data, but here we have full history)
        'completed': 'first' # Target
    })
    
    # Flatten columns
    student_features.columns = ['_'.join(col).strip() for col in student_features.columns.values]
    student_features.rename(columns={'completed_first': 'completed'}, inplace=True)
    
    # Fill NA (std can be NaN if only 1 chapter)
    student_features.fillna(0, inplace=True)
    
    # Features & Target
    X = student_features.drop(columns=['completed'])
    y = student_features['completed']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # --- Train Completion Model ---
    print("Training Completion Model...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_scaled, y_train)
    
    y_pred = clf.predict(X_test_scaled)
    print("Completion Model Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    
    # --- Save Artifacts ---
    os.makedirs("models", exist_ok=True)
    joblib.dump(clf, os.path.join("models", "completion_model.pkl"))
    joblib.dump(scaler, os.path.join("models", "scaler.pkl"))
    
    # --- Difficulty Analysis (Simple Statistical Model) ---
    # We don't "train" a model per se, but we compute difficulty stats to be used by the tool
    print("Computing Chapter Difficulty...")
    difficulty_stats = df.groupby(['course_id', 'chapter_id']).agg({
        'score': 'mean',
        'time_spent': 'mean'
    }).reset_index()
    
    # Normalize to create a "difficulty score" (Lower score + Higher time = Harder)
    # Simple heuristic: Difficulty = (MaxScore - AvgScore) + (AvgTime / MaxTime * 100)
    # This is just for demonstration of "Insight Generation"
    difficulty_stats['difficulty_score'] = (100 - difficulty_stats['score']) + (difficulty_stats['time_spent'] / difficulty_stats['time_spent'].max() * 50)
    
    # Save this "model" (lookup table)
    difficulty_stats.to_csv(os.path.join("models", "difficulty_stats.csv"), index=False)
    
    print("Models and artifacts saved to models/")

if __name__ == "__main__":
    train_models()
